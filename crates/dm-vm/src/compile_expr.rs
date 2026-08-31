//! Expression grammar, parsing, and portable-bytecode emission.
//!
//! Owns the `Expression` AST, the recursive-descent `ExpressionParser`, and
//! the emitter that lowers expressions (including call argument lists,
//! assignments, and mutations) onto the instruction stream. Statement
//! lowering in the sibling `compile_stmt` module drives the public entry
//! points here.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use dm_core::DmNumberBits;
use dm_lexer::{SpannedToken, TokenKind, lex};
use dm_value::{FieldName, TypePath};

use crate::builtins::standard_builtin_arity;
use crate::bytecode::{
    InitializerBinding, Instruction, ListEntryKind, ProcedureId, TypePredicateKind,
};
use crate::{
    CompileError, TEXT_MACRO_A, TEXT_MACRO_A_UPPER, TEXT_MACRO_IMPROPER, TEXT_MACRO_OBJECT,
    TEXT_MACRO_ORDINAL, TEXT_MACRO_PLURAL, TEXT_MACRO_POSSESSIVE, TEXT_MACRO_POSSESSIVE_ADJECTIVE,
    TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER, TEXT_MACRO_POSSESSIVE_UPPER, TEXT_MACRO_PROPER,
    TEXT_MACRO_REFLEXIVE, TEXT_MACRO_ROMAN, TEXT_MACRO_ROMAN_UPPER, TEXT_MACRO_SUBJECT,
    TEXT_MACRO_SUBJECT_UPPER, TEXT_MACRO_THE, TEXT_MACRO_THE_UPPER,
};

use crate::compile::compile_error;
use crate::compile_stmt::{
    LocalTable, compound_instruction, compound_list_index_operator, expression_static_type,
    patch_jump,
};

pub(crate) fn to_local_index(index: usize) -> Result<u16, CompileError> {
    u16::try_from(index).map_err(|_| compile_error("procedure has more than 65536 locals"))
}

pub(crate) fn compile_expression(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let expression = ExpressionParser::new(tokens).parse()?;
    emit_expression(&expression, locals, instructions, procedures)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Expression {
    Null,
    Number(DmNumberBits),
    Text(String),
    File(String),
    TypePath(TypePath),
    ModifiedTypePath {
        base: TypePath,
        overrides: Vec<(FieldName, Self)>,
    },
    New {
        type_path: Option<Box<Self>>,
        arguments: Vec<Self>,
        overrides: Vec<(FieldName, Self)>,
    },
    Regex {
        arguments: Vec<Self>,
    },
    MutableAppearance {
        arguments: Vec<Self>,
    },
    Matrix {
        arguments: Vec<Self>,
    },
    Vector {
        arguments: Vec<Self>,
    },
    ReplaceText {
        arguments: Vec<Self>,
        exact: bool,
        character_indices: bool,
    },
    CopyText {
        arguments: Vec<Self>,
        character_indices: bool,
    },
    StandardBuiltin {
        name: String,
        arguments: Vec<Self>,
    },
    NativeSrcMethod {
        name: String,
        arguments: Vec<Self>,
    },
    ExternalCall {
        library: Box<Self>,
        function: Box<Self>,
        arguments: Vec<Self>,
    },
    Animate {
        arguments: Vec<(Option<String>, Self)>,
    },
    Filter {
        arguments: Vec<(Option<String>, Self)>,
    },
    Crash(Box<Self>),
    Sleep(Box<Self>),
    Initial(Box<Self>),
    Block {
        arguments: Vec<Self>,
    },
    Rand {
        arguments: Vec<Self>,
    },
    Roll {
        arguments: Vec<Self>,
    },
    Pick {
        entries: Vec<(Option<Self>, Self)>,
    },
    Prob(Box<Self>),
    Round {
        arguments: Vec<Self>,
    },
    Length {
        value: Box<Self>,
    },
    Ref {
        value: Box<Self>,
    },
    GetStep {
        source: Box<Self>,
        direction: Box<Self>,
    },
    GetStepTowards {
        source: Box<Self>,
        target: Box<Self>,
    },
    Range {
        arguments: Vec<Self>,
    },
    TypesOf {
        arguments: Vec<Self>,
    },
    HasCall {
        receiver: Box<Self>,
        selector: Box<Self>,
    },
    TypePredicate {
        kind: TypePredicateKind,
        arguments: Vec<Self>,
    },
    Local(String),
    Src,
    Usr,
    Caller,
    World,
    GlobalNamespace,
    Field {
        receiver: Box<Self>,
        name: FieldName,
    },
    SafeField {
        receiver: Box<Self>,
        name: FieldName,
    },
    GlobalField(FieldName),
    Result,
    Call {
        procedure: String,
        arguments: Vec<Self>,
    },
    NamedArgument {
        name: String,
        value: Box<Self>,
    },
    /// A list expansion used only in an enclosing call or constructor
    /// argument list (`target(arglist(values))`).
    ArgList(Box<Self>),
    Locate {
        arguments: Vec<Self>,
    },
    LocateIn {
        arguments: Vec<Self>,
        container: Box<Self>,
    },
    CurrentCall {
        arguments: Option<Vec<Self>>,
    },
    ParentCall {
        arguments: Option<Vec<Self>>,
    },
    DynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
        null_receiver_is_global: bool,
    },
    SafeDynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    List(Vec<ListExpressionEntry>),
    AssociativeList(Vec<ListExpressionEntry>),
    Index {
        list: Box<Self>,
        index: Box<Self>,
    },
    SafeIndex {
        list: Box<Self>,
        index: Box<Self>,
    },
    Unary {
        operator: String,
        operand: Box<Self>,
    },
    Mutation {
        target: Box<Self>,
        delta: i8,
        prefix: bool,
    },
    Binary {
        operator: String,
        left: Box<Self>,
        right: Box<Self>,
    },
    Conditional {
        condition: Box<Self>,
        when_true: Box<Self>,
        when_false: Box<Self>,
    },
    LogicalOrAssignment {
        target: Box<Self>,
        value: Box<Self>,
    },
    Assignment {
        target: Box<Self>,
        operator: String,
        value: Box<Self>,
    },
}

fn expression_null_propagates(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::SafeField { .. }
            | Expression::SafeIndex { .. }
            | Expression::SafeDynamicCall { .. }
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ListExpressionEntry {
    Positional(Expression),
    Associative { key: Expression, value: Expression },
}

/// Numeric constants supplied by the BYOND language rather than by project
/// source. Keep this deliberately finite: an unrecognised identifier must
/// continue through ordinary local/field resolution and retain its useful
/// diagnostic instead of silently becoming a number.
fn dm_builtin_text_constant(identifier: &str) -> Option<&'static str> {
    match identifier {
        "UNIX" => Some("UNIX"),
        "MS_WINDOWS" => Some("MS Windows"),
        "MALE" => Some("male"),
        "FEMALE" => Some("female"),
        "NEUTER" => Some("neuter"),
        "PLURAL" => Some("plural"),
        _ => None,
    }
}

pub(crate) fn dm_builtin_numeric_constant(identifier: &str) -> Option<f32> {
    match identifier {
        "FALSE"
        | "BLEND_DEFAULT"
        | "MATRIX_COPY"
        | "MOB_PERSPECTIVE"
        | "TOPDOWN_MAP"
        | "LINEAR_EASING"
        | "COLORSPACE_RGB"
        | "MOUSE_INACTIVE_POINTER"
        | "NO_STEPS"
        | "PROFILE_START"
        | "PROFILE_REFRESH"
        | "FILTER_COLOR_RGB"
        | "UNIFORM_RAND"
        | "ICON_ADD" => Some(0.0),
        "FLOAT_LAYER" => Some(-1.0),
        "TRUE"
        | "SOUND_STREAM"
        | "NORMAL_RAND"
        | "MASK_INVERSE"
        | "FILTER_OVERLAY"
        | "FILTER_COLOR_HSV"
        | "OUTLINE_SHARP"
        | "WAVE_SIDEWAYS"
        | "ICON_SUBTRACT"
        | "BLEND_OVERLAY"
        | "KEEP_TOGETHER"
        | "NORTH"
        | "EYE_PERSPECTIVE"
        | "AREA_LAYER"
        | "SINE_EASING"
        | "ANIMATION_END_NOW"
        | "COLORSPACE_HSV"
        | "VIS_INHERIT_ICON"
        | "MOUSE_ACTIVE_POINTER"
        | "FORWARD_STEPS"
        | "BLIND"
        | "PROFILE_STOP" => Some(1.0),
        "CONTROL_FREAK_SKIN" => Some(1.0),
        "CONTROL_FREAK_MACROS" => Some(2.0),
        "JSON_PRETTY_PRINT" => Some(1.0),
        "BLEND_ADD"
        | "LINEAR_RAND"
        | "MASK_SWAP"
        | "FILTER_UNDERLAY"
        | "FILTER_COLOR_HSL"
        | "OUTLINE_SQUARE"
        | "WAVE_BOUNDED"
        | "KEEP_APART"
        | "SOUTH"
        | "EDGE_PERSPECTIVE"
        | "TURF_LAYER"
        | "CIRCULAR_EASING"
        | "ANIMATION_LINEAR_TRANSFORM"
        | "COLORSPACE_HSL"
        | "VIS_INHERIT_ICON_STATE"
        | "SLIDE_STEPS"
        | "PROFILE_CLEAR"
        | "PROFILE_RESTART"
        | "ICON_MULTIPLY" => Some(2.0),
        "BLEND_SUBTRACT" | "SQUARE_RAND" | "FILTER_COLOR_HCY" | "OBJ_LAYER" | "CUBIC_EASING"
        | "COLORSPACE_HCY" | "MOUSE_DRAG_POINTER" | "SYNC_STEPS" | "ICON_OVERLAY" => Some(3.0),
        "BLEND_MULTIPLY" | "LONG_GLIDE" | "EAST" | "MATRIX_INVERT" | "MOB_LAYER"
        | "BOUNCE_EASING" | "ANIMATION_PARALLEL" | "VIS_INHERIT_DIR" | "MOUSE_DROP_POINTER"
        | "SEE_MOBS" | "SEEMOBS" | "PROFILE_AVERAGE" => Some(4.0),
        "SOUND_UPDATE" => Some(16.0),
        "BLEND_INSET_OVERLAY"
        | "NORTHEAST"
        | "MATRIX_ROTATE"
        | "FLY_LAYER"
        | "ELASTIC_EASING"
        | "MOUSE_ARROW_POINTER"
        | "ICON_OR" => Some(5.0),
        "SOUTHEAST"
        | "MATRIX_SCALE"
        | "BACK_EASING"
        | "MOUSE_CROSSHAIRS_POINTER"
        | "ICON_UNDERLAY" => Some(6.0),
        "MATRIX_TRANSLATE" | "QUAD_EASING" | "MOUSE_HAND_POINTER" => Some(7.0),
        "WEST" | "RESET_TRANSFORM" | "JUMP_EASING" | "ANIMATION_SLICE" | "VIS_INHERIT_LAYER"
        | "SEE_OBJS" | "SEEOBJS" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" | "RESET_COLOR" | "ANIMATION_END_LOOP" | "VIS_INHERIT_PLANE" | "SEE_TURFS"
        | "SEETURFS" => Some(16.0),
        "DOWN" | "RESET_ALPHA" | "VIS_INHERIT_ID" | "SEE_SELF" => Some(32.0),
        // Appearance flags are BYOND bitflags. Keep the complete contiguous
        // built-in flag family here rather than teaching project code about
        // individual flags as each one is encountered.
        // These make an overlay/image ignore the corresponding value
        // inherited from its parent.
        "PIXEL_SCALE" | "EASE_IN" | "VIS_UNDERLAY" | "SEE_INFRA" => Some(64.0),
        "TILE_BOUND" | "MATRIX_MODIFY" | "EASE_OUT" | "VIS_HIDE" => Some(128.0),
        "INHERIT_ID" | "ANIMATION_RELATIVE" | "SEE_PIXELS" => Some(256.0),
        "NO_CLIENT_COLOR" | "ANIMATION_CONTINUE" | "SEE_THRU" => Some(512.0),
        "RESET_CONTENTS" | "SEE_BLACKNESS" => Some(1024.0),
        "PLANE_MASTER" => Some(2048.0),
        "PASS_MOUSE" => Some(4096.0),
        "TILE_MOVER" => Some(8192.0),
        "EFFECTS_LAYER" => Some(5000.0),
        "TOPDOWN_LAYER" => Some(10000.0),
        "BACKGROUND_LAYER" => Some(20000.0),
        "FLOAT_PLANE" => Some(-32767.0),
        "TILED_ICON_MAP" => Some(32768.0),
        _ => None,
    }
}

pub(crate) struct ExpressionParser<'a> {
    tokens: &'a [SpannedToken],
    pub(crate) index: usize,
    /// While parsing the true arm of `?:`, a bare colon terminates that arm
    /// instead of selecting a dynamic field.  Outside that one context DM's
    /// `datum:field` syntax remains a normal postfix operation, including in
    /// the false arm (`condition ? datum : datum:type`).
    conditional_true_arm: bool,
}

impl<'a> ExpressionParser<'a> {
    pub(crate) const fn new(tokens: &'a [SpannedToken]) -> Self {
        Self {
            tokens,
            index: 0,
            conditional_true_arm: false,
        }
    }

    pub(crate) fn parse(mut self) -> Result<Expression, CompileError> {
        let expression = self.parse_assignment()?;
        if self.index != self.tokens.len() {
            return Err(compile_error(format!(
                "unexpected token {:?} in expression",
                self.tokens[self.index].kind
            )));
        }
        Ok(expression)
    }

    /// Parses right-associative assignment expressions. DM permits an
    /// assignment anywhere an expression is accepted, for example
    /// `(GLOB.initialized = TRUE)` in a macro expansion.
    fn parse_assignment(&mut self) -> Result<Expression, CompileError> {
        let target = self.parse_conditional()?;
        let Some(TokenKind::Operator(operator)) =
            self.tokens.get(self.index).map(|token| &token.kind)
        else {
            return Ok(target);
        };
        if !matches!(
            operator.as_str(),
            "=" | ":="
                | "+="
                | "-="
                | "*="
                | "/="
                | "%="
                | "%%="
                | "&="
                | "|="
                | "^="
                | "<<="
                | ">>="
                | "&&="
                | "||="
        ) {
            return Ok(target);
        }
        let operator = if operator == ":=" {
            "=".to_owned()
        } else {
            operator.clone()
        };
        self.index += 1;
        let value = self.parse_assignment()?;
        if operator == "||=" {
            return Ok(Expression::LogicalOrAssignment {
                target: Box::new(target),
                value: Box::new(value),
            });
        }
        if operator == "&&=" {
            let assignment = Expression::Assignment {
                target: Box::new(target.clone()),
                operator: "=".to_owned(),
                value: Box::new(value),
            };
            return Ok(Expression::Conditional {
                condition: Box::new(target.clone()),
                when_true: Box::new(assignment),
                when_false: Box::new(target),
            });
        }
        Ok(Expression::Assignment {
            target: Box::new(target),
            operator,
            value: Box::new(value),
        })
    }

    /// Parses DM's right-associative `condition ? when_true : when_false`
    /// expression.  It deliberately sits below every binary operator, so a
    /// condition such as `a || b ? c : d` is parsed as expected.
    fn parse_conditional(&mut self) -> Result<Expression, CompileError> {
        let condition = self.parse_binary(1)?;
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Operator(operator)) if operator == "?"
        ) {
            return Ok(condition);
        }
        self.index += 1;
        let enclosing_true_arm = self.conditional_true_arm;
        self.conditional_true_arm = true;
        let when_true = self.parse_assignment()?;
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) if operator == ":" => self.index += 1,
            _ => return Err(compile_error("expected ':' in conditional expression")),
        }
        // The false arm is still inside an enclosing true arm, if there is
        // one.  In `a ? b ? c : d : e`, that outer colon must terminate the
        // nested expression rather than becoming dynamic access `d:e`.
        self.conditional_true_arm = enclosing_true_arm;
        let when_false = self.parse_assignment()?;
        Ok(Expression::Conditional {
            condition: Box::new(condition),
            when_true: Box::new(when_true),
            when_false: Box::new(when_false),
        })
    }

    fn parse_binary(&mut self, minimum_precedence: u8) -> Result<Expression, CompileError> {
        let mut left = self.parse_unary()?;
        while let Some(operator) = self.current_operator() {
            let Some(precedence) = binary_precedence(operator) else {
                break;
            };
            if precedence < minimum_precedence {
                break;
            }
            let operator = operator.to_owned();
            self.index += 1;
            let right_precedence = if operator == "**" {
                precedence
            } else {
                precedence + 1
            };
            let right = self.parse_binary(right_precedence)?;
            left = if operator == "in" {
                // `value in lower to upper` is BYOND's inclusive range
                // predicate. `to` is a keyword delimiter rather than a
                // general arithmetic operator, so lower it directly to the
                // two comparisons while the left operand is still available.
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Identifier(keyword)) if keyword == "to"
                ) {
                    self.index += 1;
                    let upper = self.parse_binary(right_precedence)?;
                    Expression::Binary {
                        operator: "&&".to_owned(),
                        left: Box::new(Expression::Binary {
                            operator: ">=".to_owned(),
                            left: Box::new(left.clone()),
                            right: Box::new(right),
                        }),
                        right: Box::new(Expression::Binary {
                            operator: "<=".to_owned(),
                            left: Box::new(left),
                            right: Box::new(upper),
                        }),
                    }
                } else {
                    match left {
                        Expression::Locate { arguments } => Expression::LocateIn {
                            arguments,
                            container: Box::new(right),
                        },
                        left => Expression::Binary {
                            operator,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                    }
                }
            } else {
                Expression::Binary {
                    operator,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, CompileError> {
        // Prefix mutation is an expression in DM, not merely a statement:
        // `values[++i]` first updates i, then uses the new value. Reuse the
        // assignment lowering so every assignable target retains its normal
        // single-evaluation behavior.
        if let Some(operator @ ("++" | "--")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            let target = self.parse_unary()?;
            return Ok(Expression::Mutation {
                target: Box::new(target),
                delta: if operator == "++" { 1 } else { -1 },
                prefix: true,
            });
        }
        if let Some(operator @ ("!" | "+" | "-" | "~" | "&" | "*")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            return Ok(Expression::Unary {
                operator,
                operand: Box::new(self.parse_unary()?),
            });
        }
        let mut expression = self.parse_primary()?;
        loop {
            let safe_list_index = matches!(self.current_operator(), Some("?["));
            let starts_list_index = matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('['))
            ) || safe_list_index;
            if starts_list_index {
                self.index += 1;
                // An index is a full DM expression. In particular, ternaries
                // and assignments are legal here (`values[flag ? a : b]`).
                // Parsing only the binary-precedence layer left the `?` in
                // front of the closing bracket and produced a misleading
                // "expected ']'" diagnostic.
                let index = self.parse_assignment()?;
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(']'))
                ) {
                    return Err(compile_error("expected ']' after list index"));
                }
                self.index += 1;
                expression = if safe_list_index || expression_null_propagates(&expression) {
                    Expression::SafeIndex {
                        list: Box::new(expression),
                        index: Box::new(index),
                    }
                } else {
                    Expression::Index {
                        list: Box::new(expression),
                        index: Box::new(index),
                    }
                };
                continue;
            }
            if matches!(self.current_operator(), Some("::")) {
                self.index += 1;
                let Some(TokenKind::Identifier(qualified)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected identifier after '::'"));
                };
                let qualified = qualified.clone();
                self.index += 1;
                if qualified == "name"
                    && let Expression::TypePath(path) = &expression
                    && let Some((_, procedure_name)) = path.as_str().rsplit_once("/proc/")
                {
                    expression = Expression::Text(procedure_name.to_owned());
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    expression = Expression::Call {
                        procedure: qualified,
                        arguments: self.parse_call_arguments()?,
                    };
                } else {
                    let name = FieldName::parse(&qualified)
                        .map_err(|error| compile_error(error.to_string()))?;
                    expression = Expression::Initial(Box::new(Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }));
                }
                continue;
            }
            if matches!(self.current_operator(), Some("." | "?." | "?:"))
                || (matches!(self.current_operator(), Some(":"))
                    && (!self.conditional_true_arm
                        || (self.colon_member_is_lexically_attached()
                            && self.conditional_true_arm_has_later_colon()))
                    && matches!(
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                        Some(TokenKind::Identifier(_))
                    ))
            {
                let safe_member = matches!(self.current_operator(), Some("?." | "?:"));
                self.index += 1;
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected a field name after member access"));
                };
                let name =
                    FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
                self.index += 1;
                let propagate_null = safe_member || expression_null_propagates(&expression);
                expression = if matches!(expression, Expression::GlobalNamespace) {
                    Expression::GlobalField(name)
                } else if propagate_null {
                    Expression::SafeField {
                        receiver: Box::new(expression),
                        name,
                    }
                } else {
                    Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }
                };
                continue;
            }
            // `input(...) as null|anything in choices` is prompt metadata, not
            // a cast. Retain it in the internal builtin selector and append
            // the evaluated choice list so a connected client can display the
            // correct modal while headless execution keeps its default path.
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Identifier(keyword)) if keyword == "as"
            ) {
                self.index += 1;
                let mut prompt_types = Vec::new();
                loop {
                    match self.tokens.get(self.index).map(|token| &token.kind) {
                        Some(TokenKind::Identifier(keyword)) if keyword == "in" => break,
                        Some(TokenKind::Punctuation(',' | ')' | ']' | '}')) | None => {
                            if let Expression::StandardBuiltin { name, .. } = &mut expression
                                && name == "input"
                            {
                                *name = format!("input@{}", prompt_types.join("+"));
                            }
                            return Ok(expression);
                        }
                        Some(TokenKind::Identifier(prompt_type)) => {
                            prompt_types.push(prompt_type.to_ascii_lowercase());
                            self.index += 1;
                        }
                        _ => self.index += 1,
                    }
                }
                self.index += 1;
                let choices = self.parse_assignment()?;
                if let Expression::StandardBuiltin { name, arguments } = &mut expression
                    && name == "input"
                {
                    prompt_types.push("list".to_owned());
                    *name = format!("input@{}", prompt_types.join("+"));
                    arguments.push(choices);
                }
                continue;
            }
            // A datum procedure call is a postfix operation in DM.  The
            // regular `name(...)` arm in `parse_primary` handles static
            // calls, while `receiver.name(...)` must retain both the datum
            // receiver and its dynamically-selected procedure name.  This
            // occurs extensively in lifecycle code after macro expansion
            // (for example signal dispatch helpers).
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('('))
            ) {
                expression = match expression {
                    Expression::Field { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::DynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                            null_receiver_is_global: false,
                        }
                    }
                    Expression::SafeField { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::SafeDynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                        }
                    }
                    // A second argument list invokes the procedure selector
                    // produced by the preceding expression.  DreamMaker uses
                    // this for `call_ext(library, function)(arguments)` as
                    // well as ordinary `call(...)(...)` selectors.
                    other => Expression::DynamicCall {
                        target: Box::new(Expression::Null),
                        procedure: Box::new(other),
                        arguments: self.parse_call_arguments()?,
                        null_receiver_is_global: true,
                    },
                };
                continue;
            }
            if let Some(operator @ ("++" | "--")) = self.current_operator() {
                let delta = if operator == "++" { 1 } else { -1 };
                self.index += 1;
                expression = Expression::Mutation {
                    target: Box::new(expression),
                    delta,
                    prefix: false,
                };
                continue;
            }
            break;
        }
        Ok(expression)
    }

    fn colon_member_is_lexically_attached(&self) -> bool {
        let Some(colon) = self.tokens.get(self.index) else {
            return false;
        };
        let Some(name) = self.tokens.get(self.index + 1) else {
            return false;
        };
        colon.span.end == name.span.start
    }

    /// Inside the true arm of `?:`, an attached `:name` is ambiguous with the
    /// ternary separator (`condition ? value:null`). It can only be dynamic
    /// member access when another colon remains to terminate the conditional
    /// (`condition ? datum:field : fallback`). That delimiter can be outside
    /// grouping which began before the member, as in a macro-expanded
    /// `condition ? list[(inner ? value : value:type)] : fallback`.
    fn conditional_true_arm_has_later_colon(&self) -> bool {
        for token in self.tokens.iter().skip(self.index + 2) {
            if matches!(&token.kind, TokenKind::Operator(operator) if operator == ":") {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_lines)]
    fn parse_primary(&mut self) -> Result<Expression, CompileError> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| compile_error("expected an expression"))?;
        self.index += 1;
        match &token.kind {
            // Type paths are expression values in DM: `/obj/item/tool` is
            // distinct from text and is accepted by builtins such as
            // `istype`, `ispath`, and `new`. The lexer exposes every slash as
            // an operator, so consume the complete slash-delimited sequence
            // here before ordinary binary division is considered.
            TokenKind::Operator(operator) if operator == "/" => {
                let mut path = String::new();
                loop {
                    let Some(TokenKind::Identifier(segment)) =
                        self.tokens.get(self.index).map(|token| &token.kind)
                    else {
                        // BYOND accepts a canonical type path with a trailing
                        // slash (commonly used as an associative-list key).
                        // The slash has already been consumed; canonicalize it
                        // away once at least one real segment was collected.
                        if !path.is_empty() {
                            break;
                        }
                        return Err(compile_error("expected a type path segment after '/'"));
                    };
                    path.push('/');
                    path.push_str(segment);
                    self.index += 1;
                    if !matches!(self.current_operator(), Some("/")) {
                        break;
                    }
                    self.index += 1;
                }
                let base =
                    TypePath::parse(&path).map_err(|error| compile_error(error.to_string()))?;
                let overrides = self.parse_modified_type_overrides()?;
                if overrides.is_empty() {
                    Ok(Expression::TypePath(base))
                } else {
                    Ok(Expression::ModifiedTypePath { base, overrides })
                }
            }
            TokenKind::Operator(operator)
                if operator == ".."
                    && matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::ParentCall {
                    arguments: if arguments.is_empty() {
                        None
                    } else {
                        Some(arguments)
                    },
                })
            }
            TokenKind::Operator(operator)
                if operator == "."
                    && matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::CurrentCall {
                    arguments: if arguments.is_empty() {
                        None
                    } else {
                        Some(arguments)
                    },
                })
            }
            TokenKind::Operator(operator) if operator == "." => Ok(Expression::Result),
            TokenKind::Number(spelling) => parse_number(spelling).map(Expression::Number),
            // Resource literals are first-class file values. Keep them
            // distinct from ordinary text so BYOND's `isfile()` contract is
            // observable by project code such as the runtime DMM reader.
            TokenKind::String(text) => parse_interpolated_string(text),
            TokenKind::RawString(text) | TokenKind::TextBlock(text) => {
                Ok(Expression::Text(text.clone()))
            }
            TokenKind::Resource(text) => {
                let normalized = text.replace('\\', "/");
                Ok(Expression::File(
                    normalized
                        .strip_prefix("./")
                        .unwrap_or(&normalized)
                        .to_owned(),
                ))
            }
            TokenKind::Identifier(identifier) if identifier == "null" => Ok(Expression::Null),
            TokenKind::Identifier(identifier)
                if let Some(value) = dm_builtin_numeric_constant(identifier) =>
            {
                Ok(Expression::Number(DmNumberBits::from_f32(value)))
            }
            TokenKind::Identifier(identifier)
                if let Some(value) = dm_builtin_text_constant(identifier) =>
            {
                Ok(Expression::Text(value.to_owned()))
            }
            TokenKind::Operator(operator) if operator == "::" => {
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected global identifier after '::'"));
                };
                let name = name.clone();
                self.index += 1;
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::Call {
                        procedure: name,
                        arguments: self.parse_call_arguments()?,
                    })
                } else {
                    FieldName::parse(&name)
                        .map(Expression::GlobalField)
                        .map_err(|error| compile_error(error.to_string()))
                }
            }
            TokenKind::Identifier(identifier) if identifier == "src" => Ok(Expression::Src),
            TokenKind::Identifier(identifier) if identifier == "usr" => Ok(Expression::Usr),
            TokenKind::Identifier(identifier) if identifier == "caller" => Ok(Expression::Caller),
            TokenKind::Identifier(identifier) if identifier == "world" => Ok(Expression::World),
            TokenKind::Identifier(identifier) if identifier == "locs" => Ok(Expression::Field {
                receiver: Box::new(Expression::Src),
                name: FieldName::parse("locs").expect("built-in locs field name is valid"),
            }),
            TokenKind::Identifier(identifier) if identifier == "vars" => Ok(Expression::Field {
                receiver: Box::new(Expression::Src),
                name: FieldName::parse("vars").expect("built-in vars field name is valid"),
            }),
            // Only lowercase `global` is BYOND's built-in namespace. `GLOB`
            // in SS13 codebases is an ordinary declared global datum.
            TokenKind::Identifier(identifier) if identifier == "global" => {
                Ok(Expression::GlobalNamespace)
            }
            TokenKind::Identifier(identifier) if matches!(self.tokens.get(self.index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "::") =>
            {
                let mut qualifiers = Vec::new();
                let mut next_token = self.tokens.get(self.index).map(|token| &token.kind);
                while let Some(TokenKind::Operator(operator)) = next_token {
                    if operator != "::" {
                        break;
                    }
                    self.index += 1;
                    let token = self
                        .tokens
                        .get(self.index)
                        .ok_or_else(|| compile_error("expected namespace qualifier after '::'"))?;
                    let TokenKind::Identifier(qualified) = &token.kind else {
                        return Err(compile_error("expected identifier after '::'"));
                    };
                    qualifiers.push(qualified.clone());
                    self.index += 1;
                    next_token = self.tokens.get(self.index).map(|token| &token.kind);
                }

                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    let arguments = self.parse_call_arguments()?;
                    Ok(Expression::Call {
                        procedure: qualifiers
                            .last()
                            .expect("namespace chain has a qualifier")
                            .clone(),
                        arguments,
                    })
                } else {
                    let mut receiver = Expression::Local(identifier.clone());
                    for qualifier in qualifiers {
                        let name = FieldName::parse(&qualifier)
                            .map_err(|error| compile_error(error.to_string()))?;
                        receiver = Expression::Initial(Box::new(Expression::Field {
                            receiver: Box::new(receiver),
                            name,
                        }));
                    }
                    Ok(receiver)
                }
            }
            TokenKind::Identifier(identifier) if identifier == "new" => {
                // `new /path(args)` is the common explicit form.  An
                // unqualified `new(args)` constructs the current datum type.
                // Keep the constructor arguments in the AST even though the
                // headless VM currently only establishes object identity.
                if matches!(self.current_operator(), Some("/")) {
                    let type_path = self.parse_primary()?;
                    let overrides = self.parse_modified_type_overrides()?;
                    let arguments = if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) {
                        self.parse_call_arguments()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expression::New {
                        type_path: Some(Box::new(type_path)),
                        arguments,
                        overrides,
                    })
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: self.parse_call_arguments()?,
                        overrides: Vec::new(),
                    })
                } else if let Some(TokenKind::Identifier(type_name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                {
                    // DM also permits a runtime type expression, for example
                    // `new starting_organ(src)`.  This is distinct from
                    // unqualified `new(...)`: the identifier is the type to
                    // instantiate, not a constructor argument.
                    // Do not delegate this to `parse_unary`: its ordinary
                    // identifier rule interprets the following `(` as a
                    // static procedure call. Here it belongs to `new`.
                    let mut type_path = if type_name == "src" {
                        Expression::Src
                    } else {
                        Expression::Local(type_name.clone())
                    };
                    self.index += 1;
                    while matches!(self.current_operator(), Some(".")) {
                        self.index += 1;
                        let Some(TokenKind::Identifier(field)) =
                            self.tokens.get(self.index).map(|token| &token.kind)
                        else {
                            return Err(compile_error(
                                "runtime new type field access requires an identifier",
                            ));
                        };
                        type_path = Expression::Field {
                            receiver: Box::new(type_path),
                            name: FieldName::parse(field)
                                .map_err(|error| compile_error(error.to_string()))?,
                        };
                        self.index += 1;
                    }
                    let arguments = if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) {
                        self.parse_call_arguments()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expression::New {
                        type_path: Some(Box::new(type_path)),
                        arguments,
                        overrides: Vec::new(),
                    })
                } else {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: Vec::new(),
                        overrides: Vec::new(),
                    })
                }
            }
            TokenKind::Identifier(identifier) if identifier == "call_ext" => {
                let selectors = self.parse_call_arguments()?;
                let [library, function] = selectors.as_slice() else {
                    return Err(compile_error(
                        "call_ext requires a library and exported function selector",
                    ));
                };
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    return Err(compile_error("call_ext selector requires an argument list"));
                }
                Ok(Expression::ExternalCall {
                    library: Box::new(library.clone()),
                    function: Box::new(function.clone()),
                    arguments: self.parse_call_arguments()?,
                })
            }
            TokenKind::Identifier(identifier) if identifier == "call" => {
                let selectors = self.parse_call_arguments()?;
                let (target, procedure, null_receiver_is_global) = match selectors.as_slice() {
                    [procedure] => (Expression::Null, procedure.clone(), true),
                    [target, procedure] => (target.clone(), procedure.clone(), false),
                    _ => {
                        return Err(compile_error(
                            "call requires a procedure or a receiver and procedure",
                        ));
                    }
                };
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    return Err(compile_error("call selector requires an argument list"));
                }
                Ok(Expression::DynamicCall {
                    target: Box::new(target),
                    procedure: Box::new(procedure),
                    arguments: self.parse_call_arguments()?,
                    null_receiver_is_global,
                })
            }
            TokenKind::Identifier(identifier)
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) =>
            {
                if identifier == "CRASH" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "CRASH requires exactly one argument, received {}",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Crash(Box::new(
                        arguments.pop().expect("CRASH argument count was validated"),
                    )))
                } else if identifier == "list" {
                    Ok(Expression::List(self.parse_list_arguments()?))
                } else if identifier == "alist" {
                    Ok(Expression::AssociativeList(self.parse_list_arguments()?))
                } else if identifier == "arglist" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "arglist requires exactly one list, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::ArgList(Box::new(
                        arguments.pop().expect("argument count was validated"),
                    )))
                } else if let Some(kind) = type_predicate_kind(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    let valid_count = match kind {
                        TypePredicateKind::IsType | TypePredicateKind::IsPath => {
                            (1..=2).contains(&arguments.len())
                        }
                        // BYOND's location classifiers accept multiple values
                        // and succeed only when every supplied value matches.
                        TypePredicateKind::IsLoc
                        | TypePredicateKind::IsMovable
                        | TypePredicateKind::IsTurf => !arguments.is_empty(),
                        _ => arguments.len() == 1,
                    };
                    if !valid_count {
                        return Err(compile_error(format!(
                            "{identifier} received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::TypePredicate { kind, arguments })
                } else if identifier == "initial" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "initial requires exactly one variable reference, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Initial(Box::new(
                        arguments.pop().expect("validated initial argument"),
                    )))
                } else if identifier == "regex" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "regex requires a pattern and optional flags, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Regex { arguments })
                } else if identifier == "mutable_appearance" {
                    Ok(Expression::MutableAppearance {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if identifier == "matrix" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 6 {
                        return Err(compile_error("matrix accepts at most six arguments"));
                    }
                    Ok(Expression::Matrix { arguments })
                } else if identifier == "vector" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 3 {
                        return Err(compile_error("vector accepts at most three arguments"));
                    }
                    Ok(Expression::Vector { arguments })
                } else if let Some((exact, character_indices)) = replacetext_kind(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    if !(3..=5).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "{identifier} requires text, needle, replacement, and optional start/end; received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::ReplaceText {
                        arguments,
                        exact,
                        character_indices,
                    })
                } else if matches!(identifier.as_str(), "copytext" | "copytext_char") {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=3).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "{identifier} requires text and optional start/end; received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::CopyText {
                        arguments,
                        character_indices: identifier == "copytext_char",
                    })
                } else if identifier == "length" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "length requires exactly one argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Length {
                        value: Box::new(
                            arguments
                                .pop()
                                .expect("length argument count was validated"),
                        ),
                    })
                } else if identifier == "ref" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "ref requires exactly one argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Ref {
                        value: Box::new(arguments.pop().expect("ref argument count was validated")),
                    })
                } else if identifier == "get_step" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "get_step requires exactly an atom/turf and direction, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let direction = arguments
                        .pop()
                        .expect("get_step argument count was validated");
                    let source = arguments
                        .pop()
                        .expect("get_step argument count was validated");
                    Ok(Expression::GetStep {
                        source: Box::new(source),
                        direction: Box::new(direction),
                    })
                } else if identifier == "get_step_towards" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "get_step_towards requires exactly a source and target, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let target = arguments.pop().expect("argument count validated");
                    let source = arguments.pop().expect("argument count validated");
                    Ok(Expression::GetStepTowards {
                        source: Box::new(source),
                        target: Box::new(target),
                    })
                } else if identifier == "range" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "range requires a distance and optional center, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Range { arguments })
                } else if identifier == "block" {
                    let arguments = self.parse_call_arguments()?;
                    if !(arguments.len() == 2 || (3..=6).contains(&arguments.len())) {
                        return Err(compile_error(format!(
                            "block requires two turfs or three through six coordinates, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Block { arguments })
                } else if identifier == "typesof" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.is_empty() || arguments.len() > usize::from(u8::MAX) {
                        return Err(compile_error(format!(
                            "typesof requires between one and 255 type arguments, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::TypesOf { arguments })
                } else if identifier == "hascall" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "hascall requires a receiver and procedure selector, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let selector = arguments.pop().expect("hascall arity was validated");
                    let receiver = arguments.pop().expect("hascall arity was validated");
                    Ok(Expression::HasCall {
                        receiver: Box::new(receiver),
                        selector: Box::new(selector),
                    })
                } else if identifier == "rand" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 2 {
                        return Err(compile_error(format!(
                            "rand accepts zero, one, or two numeric bounds, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Rand { arguments })
                } else if identifier == "roll" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "roll requires dice or a dice count and side count, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Roll { arguments })
                } else if identifier == "pick" {
                    Ok(Expression::Pick {
                        entries: self.parse_pick_arguments()?,
                    })
                } else if identifier == "prob" {
                    let arguments = self.parse_call_arguments()?;
                    let [chance] = arguments.as_slice() else {
                        return Err(compile_error(format!(
                            "prob requires exactly one percentage, received {} arguments",
                            arguments.len()
                        )));
                    };
                    Ok(Expression::Prob(Box::new(chance.clone())))
                } else if identifier == "round" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "round requires a number and optional multiple, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Round { arguments })
                } else if identifier == "sleep" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "sleep requires exactly one delay, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Sleep(Box::new(
                        arguments.pop().expect("sleep argument count was validated"),
                    )))
                } else if identifier == "locate" {
                    Ok(Expression::Locate {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if identifier == "animate" {
                    Ok(Expression::Animate {
                        arguments: self.parse_named_call_arguments()?,
                    })
                } else if identifier == "filter" {
                    Ok(Expression::Filter {
                        arguments: self.parse_named_call_arguments()?,
                    })
                } else if identifier == "nameof" {
                    self.parse_nameof_expression()
                } else if matches!(
                    identifier.as_str(),
                    "MapColors"
                        | "Blend"
                        | "SetIntensity"
                        | "Scale"
                        | "Crop"
                        | "Shift"
                        | "Width"
                        | "Height"
                        | "DrawBox"
                        | "Insert"
                        | "GetPixel"
                        | "Add"
                        | "Subtract"
                        | "Multiply"
                        | "Translate"
                        | "Invert"
                        | "Turn"
                ) {
                    Ok(Expression::NativeSrcMethod {
                        name: identifier.clone(),
                        arguments: self.parse_call_arguments()?,
                    })
                } else if let Some((minimum, maximum)) = standard_builtin_arity(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() < minimum || arguments.len() > maximum {
                        return Err(compile_error(format!(
                            "{identifier} received {} arguments; expected {minimum} through {maximum}",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::StandardBuiltin {
                        name: identifier.clone(),
                        arguments,
                    })
                } else {
                    let arguments = self.parse_call_arguments()?;
                    Ok(Expression::Call {
                        procedure: identifier.clone(),
                        arguments,
                    })
                }
            }
            TokenKind::Identifier(identifier) => Ok(Expression::Local(identifier.clone())),
            TokenKind::Punctuation('(') => {
                let expression = self.parse_assignment()?;
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    Some(TokenKind::Punctuation(')')) => {
                        self.index += 1;
                        Ok(expression)
                    }
                    found => Err(compile_error(format!(
                        "expected ')' after expression; found {found:?}; next {:?}",
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                    ))),
                }
            }
            _ => Err(compile_error(format!(
                "unexpected token {:?} in expression",
                token.kind
            ))),
        }
    }

    /// Parses BYOND's compile-time `nameof(reference)` form.
    ///
    /// The argument is a reference grammar rather than an ordinary runtime
    /// expression.  In particular, tgstation uses all of these shapes:
    /// `nameof(.proc/name)`, `nameof(/datum/example.proc/name)`, and
    /// `nameof(type::field)`.  Each evaluates to the referenced member's
    /// final textual component.  Retaining that component is sufficient for
    /// headless callback and signal registration and also supports
    /// `NAMEOF_STATIC` without pretending its compile-time reference is a
    /// datum field read.
    fn parse_nameof_expression(&mut self) -> Result<Expression, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut nesting = 0_usize;
        let mut final_name = None;
        loop {
            let token = self
                .tokens
                .get(self.index)
                .ok_or_else(|| compile_error("expected ')' after nameof reference"))?;
            match &token.kind {
                TokenKind::Punctuation('(') => nesting += 1,
                TokenKind::Punctuation(')') if nesting == 0 => {
                    self.index += 1;
                    break;
                }
                TokenKind::Punctuation(')') => nesting -= 1,
                TokenKind::Identifier(name) => final_name = Some(name.clone()),
                _ => {}
            }
            self.index += 1;
        }
        final_name
            .map(Expression::Text)
            .ok_or_else(|| compile_error("nameof requires a named reference"))
    }

    pub(crate) fn parse_call_arguments(&mut self) -> Result<Vec<Expression>, CompileError> {
        Ok(self
            .parse_named_call_arguments()?
            .into_iter()
            .map(|(name, expression)| {
                name.map_or(expression.clone(), |name| Expression::NamedArgument {
                    name,
                    value: Box::new(expression),
                })
            })
            .collect())
    }

    fn parse_named_call_arguments(
        &mut self,
    ) -> Result<Vec<(Option<String>, Expression)>, CompileError> {
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ) {
            return Err(compile_error("expected '(' before call arguments"));
        }
        self.index += 1;
        let mut arguments = Vec::new();
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            loop {
                // Empty positional slots are legal DM and mean that the
                // callee's default value should be used.  The VM represents
                // an omitted interior slot as null until its call ABI carries
                // a distinct omission marker.
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(','))
                ) {
                    arguments.push((None, Expression::Null));
                    self.index += 1;
                    if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation(')'))
                    ) {
                        break;
                    }
                    continue;
                }
                // BYOND permits keyword-style call arguments, e.g.
                // `do_after(user, 4 SECONDS, target = src)`.  The current
                // execution ABI is positional, but retaining the source
                // order here is still the correct lowering for its existing
                // subset and, importantly, lets the compiler continue on to
                // report the next unsupported construct instead of rejecting
                // the call syntax itself.
                let name = match (
                    self.tokens.get(self.index).map(|token| &token.kind),
                    self.tokens.get(self.index + 1).map(|token| &token.kind),
                ) {
                    (Some(TokenKind::Identifier(name)), Some(TokenKind::Operator(operator)))
                        if operator == "=" =>
                    {
                        Some(name.clone())
                    }
                    (Some(TokenKind::String(name)), Some(TokenKind::Operator(operator)))
                        if operator == "=" =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                };
                if name.is_some() {
                    self.index += 2;
                }
                arguments.push((name, self.parse_assignment()?));
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    // DM's weighted `pick()` syntax separates a weight from
                    // its candidate with `;`, e.g. `pick(10; red, 1; blue)`.
                    // The headless call ABI is positional, so retaining both
                    // expressions is the most faithful representation it can
                    // currently carry.
                    Some(TokenKind::Punctuation(',' | ';')) => {
                        self.index += 1;
                        // DM accepts a trailing separator in a parenthesized
                        // argument list, including multiline calls.  Do not
                        // attempt to parse the closing parenthesis as the
                        // next argument expression.
                        if matches!(
                            self.tokens.get(self.index).map(|token| &token.kind),
                            Some(TokenKind::Punctuation(')'))
                        ) {
                            break;
                        }
                    }
                    Some(TokenKind::Punctuation(')')) => break,
                    _ => {
                        return Err(compile_error(format!(
                            "expected ',' or ')' after procedure argument, received {:?}",
                            self.tokens.get(self.index).map(|token| &token.kind)
                        )));
                    }
                }
            }
        }
        self.index += 1;
        Ok(arguments)
    }

    /// Parses `pick()` entries while retaining its `weight; candidate` form.
    fn parse_pick_arguments(
        &mut self,
    ) -> Result<Vec<(Option<Expression>, Expression)>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut entries = Vec::new();
        while !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            let first = self.parse_assignment()?;
            let entry = if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation(';'))
            ) {
                self.index += 1;
                (Some(first), self.parse_assignment()?)
            } else {
                (None, first)
            };
            entries.push(entry);
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(',')) => self.index += 1,
                Some(TokenKind::Punctuation(')')) => break,
                _ => return Err(compile_error("expected ',' or ')' after pick entry")),
            }
        }
        if entries.is_empty() {
            return Err(compile_error("pick requires at least one candidate"));
        }
        self.index += 1;
        Ok(entries)
    }

    fn parse_modified_type_overrides(
        &mut self,
    ) -> Result<Vec<(FieldName, Expression)>, CompileError> {
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('{'))
        ) {
            return Ok(Vec::new());
        }
        self.index += 1;
        let mut overrides = Vec::new();
        loop {
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('}'))
            ) {
                self.index += 1;
                return Ok(overrides);
            }
            let Some(TokenKind::Identifier(name)) =
                self.tokens.get(self.index).map(|token| &token.kind)
            else {
                return Err(compile_error("modified type requires a field name"));
            };
            let name = FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
            self.index += 1;
            if !matches!(self.current_operator(), Some("=")) {
                return Err(compile_error("modified type field requires '='"));
            }
            self.index += 1;
            let start = self.index;
            let mut depth = 0_usize;
            while let Some(token) = self.tokens.get(self.index) {
                match token.kind {
                    TokenKind::Punctuation('(' | '[') => depth += 1,
                    TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
                    TokenKind::Punctuation('}' | ';') if depth == 0 => break,
                    _ => {}
                }
                self.index += 1;
            }
            if start == self.index {
                return Err(compile_error("modified type field value is empty"));
            }
            let value = ExpressionParser::new(&self.tokens[start..self.index]).parse()?;
            overrides.push((name, value));
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(';')) => self.index += 1,
                Some(TokenKind::Punctuation('}')) => {}
                _ => return Err(compile_error("modified type requires ';' or '}'")),
            }
        }
    }

    fn parse_list_arguments(&mut self) -> Result<Vec<ListExpressionEntry>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut entries = Vec::new();
        while !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            // BYOND treats an omitted interior list/call argument as null,
            // while a single trailing comma contributes no extra entry.
            // Monk's species perk lists intentionally rely on this before
            // filtering nulls with list_clear_nulls().
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation(','))
            ) {
                entries.push(ListExpressionEntry::Positional(Expression::Null));
                self.index += 1;
                continue;
            }
            // The unparenthesized `=` in a list literal introduces an
            // associative entry rather than an assignment expression. A
            // parenthesized assignment still reaches `parse_assignment` via
            // primary-expression parsing.
            let key_or_value = self.parse_conditional()?;
            if matches!(self.current_operator(), Some("=")) {
                self.index += 1;
                let value = self.parse_conditional()?;
                entries.push(ListExpressionEntry::Associative {
                    key: key_or_value,
                    value,
                });
            } else {
                entries.push(ListExpressionEntry::Positional(key_or_value));
            }
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(',')) => self.index += 1,
                Some(TokenKind::Punctuation(')')) => break,
                _ => return Err(compile_error("expected ',' or ')' after list entry")),
            }
        }
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            return Err(compile_error("expected ')' after list entries"));
        }
        self.index += 1;
        Ok(entries)
    }

    fn current_operator(&self) -> Option<&str> {
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) => Some(operator),
            Some(TokenKind::Identifier(identifier)) if identifier == "in" => Some(identifier),
            _ => None,
        }
    }
}

/// Classifies BYOND's four `replacetext` builtin spellings without treating
/// them as project procedures.  The `_char` variants use character positions,
/// while `Ex` means exact (case-sensitive) matching.
fn replacetext_kind(identifier: &str) -> Option<(bool, bool)> {
    match identifier {
        "replacetext" => Some((false, false)),
        "replacetextEx" => Some((true, false)),
        "replacetext_char" => Some((false, true)),
        "replacetextEx_char" => Some((true, true)),
        _ => None,
    }
}

/// Identifies the compiler-handled BYOND value predicates.
fn type_predicate_kind(identifier: &str) -> Option<TypePredicateKind> {
    match identifier {
        "isnull" => Some(TypePredicateKind::IsNull),
        "isnum" => Some(TypePredicateKind::IsNum),
        "ispath" => Some(TypePredicateKind::IsPath),
        "islist" => Some(TypePredicateKind::IsList),
        "ismovable" => Some(TypePredicateKind::IsMovable),
        "isturf" => Some(TypePredicateKind::IsTurf),
        "isloc" => Some(TypePredicateKind::IsLoc),
        "isicon" => Some(TypePredicateKind::IsIcon),
        "istype" => Some(TypePredicateKind::IsType),
        _ => None,
    }
}

fn parse_number(spelling: &str) -> Result<DmNumberBits, CompileError> {
    let normalized = spelling.replace('_', "");
    let value = if matches!(normalized.as_str(), "1#INF" | "1.#INF") {
        f32::INFINITY
    } else if matches!(normalized.as_str(), "1#IND" | "1.#IND") {
        f32::NAN
    } else if let Some(hexadecimal) = normalized
        .strip_prefix("0x")
        .or_else(|| normalized.strip_prefix("0X"))
    {
        let integer = u32::from_str_radix(hexadecimal, 16)
            .map_err(|error| compile_error(format!("invalid number {spelling:?}: {error}")))?;
        integer
            .to_string()
            .parse::<f32>()
            .expect("every u32 decimal spelling is a valid f32")
    } else {
        normalized
            .parse::<f32>()
            .map_err(|error| compile_error(format!("invalid number {spelling:?}: {error}")))?
    };
    Ok(DmNumberBits::from_f32(value))
}

fn parse_interpolated_string(text: &str) -> Result<Expression, CompileError> {
    const ESCAPED_OPEN: char = '\u{e000}';
    const ESCAPED_CLOSE: char = '\u{e001}';
    // Protect escaped brackets before looking for interpolation holes. This
    // must consume escape pairs rather than using `str::replace`: in
    // `\\\\[value]` the first pair denotes a literal backslash and the
    // bracket still begins interpolation.
    let mut protected = String::with_capacity(text.len());
    let mut characters = text.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            protected.push(character);
            continue;
        }
        let Some(escaped) = characters.next() else {
            protected.push('\\');
            break;
        };
        match escaped {
            '[' => protected.push(ESCAPED_OPEN),
            ']' => protected.push(ESCAPED_CLOSE),
            _ => {
                protected.push('\\');
                protected.push(escaped);
            }
        }
    }
    let text = protected;
    let literal = |text: &str| {
        decode_quoted_text_fragment(&text.replace(ESCAPED_OPEN, "[").replace(ESCAPED_CLOSE, "]"))
    };
    let mut template = String::with_capacity(text.len());
    let mut interpolations = Vec::new();
    let mut cursor = 0_usize;
    while let Some(relative_open) = text[cursor..].find('[') {
        let open = cursor + relative_open;
        let Some(close) = interpolated_expression_close(&text, open + 1) else {
            break;
        };
        if text[open + 1..close].trim().is_empty() {
            cursor = close + 1;
            continue;
        }
        if open > cursor {
            template.push_str(&literal(&text[cursor..open]));
        }
        let tokens = lex(&text[open + 1..close])
            .map_err(|error| {
                compile_error(format!("invalid embedded expression: {}", error.message))
            })?
            .into_iter()
            .filter(|token| {
                !matches!(
                    token.kind,
                    TokenKind::LineStart { .. } | TokenKind::Newline | TokenKind::LineContinuation
                )
            })
            .collect::<Vec<_>>();
        template.push_str("[]");
        interpolations.push(ExpressionParser::new(&tokens).parse()?);
        cursor = close + 1;
    }
    if interpolations.is_empty() {
        return Ok(Expression::Text(literal(&text)));
    }
    if cursor < text.len() {
        template.push_str(&literal(&text[cursor..]));
    }
    let mut arguments = Vec::with_capacity(interpolations.len() + 1);
    arguments.push(Expression::Text(template));
    arguments.extend(interpolations);
    Ok(Expression::StandardBuiltin {
        name: "text".to_owned(),
        arguments,
    })
}

/// Decode escapes in an ordinary double-quoted DM string fragment. Raw
/// strings are represented by a different token kind and intentionally never
/// pass through this function.
fn decode_quoted_text_fragment(text: &str) -> String {
    let mut decoded = String::with_capacity(text.len());
    let mut cursor = 0_usize;
    while cursor < text.len() {
        let character = text[cursor..]
            .chars()
            .next()
            .expect("cursor is inside quoted text");
        cursor += character.len_utf8();
        if character != '\\' {
            decoded.push(character);
            continue;
        }
        if cursor == text.len() {
            decoded.push('\\');
            break;
        }

        // Text macros win over their single-character escape prefixes:
        // `\\the` and `\\th` are format operations, while `\\t` is a tab.
        let remaining = &text[cursor..];
        let macro_match = [
            ("improper", TEXT_MACRO_IMPROPER, true),
            ("himself", TEXT_MACRO_REFLEXIVE, false),
            ("Himself", TEXT_MACRO_REFLEXIVE, false),
            ("herself", TEXT_MACRO_REFLEXIVE, false),
            ("Herself", TEXT_MACRO_REFLEXIVE, false),
            ("proper", TEXT_MACRO_PROPER, true),
            ("Roman", TEXT_MACRO_ROMAN_UPPER, true),
            ("roman", TEXT_MACRO_ROMAN, true),
            ("Hers", TEXT_MACRO_POSSESSIVE_UPPER, false),
            ("hers", TEXT_MACRO_POSSESSIVE, false),
            ("The", TEXT_MACRO_THE_UPPER, true),
            ("the", TEXT_MACRO_THE, true),
            ("She", TEXT_MACRO_SUBJECT_UPPER, false),
            ("she", TEXT_MACRO_SUBJECT, false),
            ("His", TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER, false),
            ("his", TEXT_MACRO_POSSESSIVE_ADJECTIVE, false),
            ("him", TEXT_MACRO_OBJECT, false),
            ("An", TEXT_MACRO_A_UPPER, true),
            ("an", TEXT_MACRO_A, true),
            ("He", TEXT_MACRO_SUBJECT_UPPER, false),
            ("he", TEXT_MACRO_SUBJECT, false),
            ("th", TEXT_MACRO_ORDINAL, false),
            ("A", TEXT_MACRO_A_UPPER, true),
            ("a", TEXT_MACRO_A, true),
            ("s", TEXT_MACRO_PLURAL, false),
        ]
        .into_iter()
        .find(|(spelling, _, _)| remaining.starts_with(spelling));
        if let Some((spelling, marker, prefix)) = macro_match {
            decoded.push(marker);
            cursor += spelling.len();
            if prefix && text[cursor..].starts_with(' ') {
                cursor += 1;
            }
            continue;
        }

        let escaped = text[cursor..]
            .chars()
            .next()
            .expect("checked escaped text exists");
        cursor += escaped.len_utf8();
        match escaped {
            'n' => decoded.push('\n'),
            'r' => decoded.push('\r'),
            't' => decoded.push('\t'),
            '\\' => decoded.push('\\'),
            '"' => decoded.push('"'),
            '[' => decoded.push('['),
            ']' => decoded.push(']'),
            // BYOND has additional display-format escapes (for example
            // `\\the` and `\\proper`). Keep those intact until the text
            // formatting layer interprets them instead of silently deleting
            // their escape marker.
            other => {
                decoded.push('\\');
                decoded.push(other);
            }
        }
    }
    decoded
}

pub(crate) fn interpolated_expression_close(text: &str, start: usize) -> Option<usize> {
    let mut depth = 1usize;
    let mut cursor = start;
    let mut quote = None;
    let mut escaped = false;
    while cursor < text.len() {
        let character = text[cursor..].chars().next()?;
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                quote = None;
            }
        } else {
            match character {
                '"' | '\'' => quote = Some(character),
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(cursor);
                    }
                }
                _ => {}
            }
        }
        cursor += character.len_utf8();
    }
    None
}

const fn binary_precedence(operator: &str) -> Option<u8> {
    match operator.as_bytes() {
        b"||" => Some(1),
        b"&&" => Some(2),
        b"|" => Some(3),
        b"^" => Some(4),
        b"&" => Some(5),
        b"==" | b"!=" | b"<>" | b"~=" | b"~!" => Some(6),
        b"<<" | b">>" | b"<" | b"<=" | b">" | b">=" | b"<=>" | b"in" => Some(7),
        b"+" | b"-" => Some(8),
        b"*" | b"/" | b"%" | b"%%" => Some(9),
        b"**" => Some(10),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
/// Emits an associative-list key, preserving macro-expanded named arguments.
///
/// Macro wrappers such as `AddComponent(...)` expand named arguments into
/// `list(name = value)`. The original call grammar is no longer visible, so
/// an unbound bare name here is a textual associative key, not an assignment
/// target. Bound locals and fields retain their ordinary expression meaning.
fn emit_associative_list_key(
    key: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    // In DM's list-constructor grammar, a bare identifier to the left of `=`
    // is a named/text key even when a local, field, or global with the same
    // spelling exists. Dynamic keys use an explicit expression instead.
    if let Expression::Local(name) = key {
        instructions.push(Instruction::PushText(Arc::from(name.as_str())));
        return Ok(());
    }
    emit_expression(key, locals, instructions, procedures)
}

/// Marker used by call-like instructions to consume the count produced by
/// [`Instruction::ExpandArgumentLists`].  A source procedure cannot have
/// this many arguments, so it is unambiguous in the compact bytecode ABI.
pub(crate) const EXPANDED_ARGUMENT_COUNT: u16 = u16::MAX;

/// Emits a call argument vector, retaining BYOND's runtime `arglist()`
/// expansion semantics.  Ordinary expressions preserve the compact static
/// count; an expansion emits a small preparation instruction and returns the
/// sentinel consumed by the following call-like instruction.
fn emit_call_arguments(
    arguments: &[Expression],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<u16, CompileError> {
    let argument_count = u16::try_from(arguments.len())
        .map_err(|_| compile_error("call has more than 65535 positional arguments"))?;
    let mut expanded_indices = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        if let Expression::ArgList(value) = argument {
            expanded_indices.push(to_local_index(index)?);
            emit_expression(value, locals, instructions, procedures)?;
        } else if let Expression::NamedArgument { value, .. } = argument {
            emit_expression(value, locals, instructions, procedures)?;
        } else {
            emit_expression(argument, locals, instructions, procedures)?;
        }
    }
    if expanded_indices.is_empty() {
        Ok(argument_count)
    } else {
        instructions.push(Instruction::ExpandArgumentLists {
            argument_count,
            argument_names: arguments
                .iter()
                .map(|argument| match argument {
                    Expression::NamedArgument { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect(),
            expanded_indices,
        });
        Ok(EXPANDED_ARGUMENT_COUNT)
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn emit_expression(
    expression: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match expression {
        Expression::NamedArgument { value, .. } => {
            emit_expression(value, locals, instructions, procedures)?;
        }
        Expression::Null => instructions.push(Instruction::PushNull),
        Expression::Number(number) => instructions.push(Instruction::PushNumber(*number)),
        Expression::Text(text) => {
            instructions.push(Instruction::PushText(Arc::from(text.as_str())));
        }
        Expression::File(path) => instructions.push(Instruction::PushFile(path.clone())),
        Expression::TypePath(path) => instructions.push(Instruction::PushTypePath(path.clone())),
        Expression::ModifiedTypePath { base, overrides } => {
            instructions.push(Instruction::PushTypePath(base.clone()));
            for (_, value) in overrides {
                emit_expression(value, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeModifiedTypePath {
                fields: overrides
                    .iter()
                    .map(|(field, _)| field.clone())
                    .collect::<Vec<_>>()
                    .into(),
            });
        }
        Expression::New {
            type_path,
            arguments,
            overrides,
        } => {
            let Some(type_path) = type_path else {
                return Err(compile_error(
                    "inferred new has no statically resolved destination type",
                ));
            };
            emit_expression(type_path, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::AllocateDatum {
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
            });
            for (name, value) in overrides {
                instructions.push(Instruction::Duplicate);
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::StoreField(name.clone()));
            }
        }
        Expression::Regex { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeRegex {
                argument_count: u8::try_from(arguments.len())
                    .expect("regex argument count was validated by the parser"),
            });
        }
        Expression::MutableAppearance { arguments } => {
            // SS13 projects commonly provide `/proc/mutable_appearance` as a
            // behavior-rich helper around the engine datum. Just like qdel
            // and the other engine fallbacks, that project procedure wins.
            // This is especially important for named arguments: Monkestation's
            // human overlay path supplies `layer` and `appearance_flags`, and
            // the helper also applies its omitted defaults before returning.
            if let Some(procedure) = procedures.get("mutable_appearance").copied() {
                let argument_count =
                    emit_call_arguments(arguments, locals, instructions, procedures)?;
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                    argument_names: arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::NamedArgument { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                });
            } else {
                for argument in arguments {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
                instructions.push(Instruction::MakeMutableAppearance {
                    argument_count: u16::try_from(arguments.len()).map_err(|_| {
                        compile_error(
                            "mutable_appearance has more than 65535 constructor arguments",
                        )
                    })?,
                });
            }
        }
        Expression::Matrix { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeMatrix {
                argument_count: u8::try_from(arguments.len())
                    .expect("matrix argument count was validated"),
            });
        }
        Expression::Vector { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeVector {
                argument_count: u8::try_from(arguments.len())
                    .expect("vector argument count was validated"),
            });
        }
        Expression::ReplaceText {
            arguments,
            exact,
            character_indices,
        } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::ReplaceText {
                argument_count: u8::try_from(arguments.len())
                    .expect("replacetext argument count was validated by the parser"),
                exact: *exact,
                character_indices: *character_indices,
            });
        }
        Expression::CopyText {
            arguments,
            character_indices,
        } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::CopyText {
                argument_count: u8::try_from(arguments.len())
                    .expect("copytext argument count was validated by the parser"),
                character_indices: *character_indices,
            });
        }
        Expression::Block { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Block {
                argument_count: u8::try_from(arguments.len())
                    .expect("block argument count was validated by the parser"),
            });
        }
        Expression::Length { value } => {
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Length);
        }
        Expression::Ref { value } => {
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Ref);
        }
        Expression::GetStep { source, direction } => {
            emit_expression(source, locals, instructions, procedures)?;
            emit_expression(direction, locals, instructions, procedures)?;
            instructions.push(Instruction::GetStep);
        }
        Expression::GetStepTowards { source, target } => {
            emit_expression(source, locals, instructions, procedures)?;
            emit_expression(target, locals, instructions, procedures)?;
            instructions.push(Instruction::GetStepTowards);
        }
        Expression::Range { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Range {
                argument_count: u8::try_from(arguments.len())
                    .expect("range argument count was validated by the parser"),
            });
        }
        Expression::TypesOf { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::TypesOf {
                argument_count: u8::try_from(arguments.len())
                    .expect("typesof argument count was validated by the parser"),
            });
        }
        Expression::HasCall { receiver, selector } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            emit_expression(selector, locals, instructions, procedures)?;
            instructions.push(Instruction::HasCall);
        }
        Expression::Rand { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Rand {
                argument_count: u8::try_from(arguments.len())
                    .expect("rand argument count was validated by the parser"),
            });
        }
        Expression::Roll { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Roll {
                argument_count: u8::try_from(arguments.len())
                    .expect("roll argument count was validated by the parser"),
            });
        }
        Expression::Pick { entries } => {
            if let [(None, Expression::ArgList(value))] = entries.as_slice() {
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::ExpandArgumentLists {
                    argument_count: 1,
                    argument_names: vec![None],
                    expanded_indices: vec![0],
                });
                instructions.push(Instruction::PickExpandedArguments);
                return Ok(());
            }
            let mut weighted = Vec::with_capacity(entries.len());
            for (weight, candidate) in entries {
                weighted.push(weight.is_some());
                if let Some(weight) = weight {
                    emit_expression(weight, locals, instructions, procedures)?;
                }
                emit_expression(candidate, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Pick { weighted });
        }
        Expression::Prob(chance) => {
            emit_expression(chance, locals, instructions, procedures)?;
            instructions.push(Instruction::Prob);
        }
        Expression::Round { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Round {
                argument_count: u8::try_from(arguments.len())
                    .expect("round argument count was validated by the parser"),
            });
        }
        Expression::TypePredicate { kind, arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            let inferred_type = (*kind == TypePredicateKind::IsType && arguments.len() == 1)
                .then(|| expression_static_type(&arguments[0], locals))
                .flatten();
            let argument_count = arguments.len() + usize::from(inferred_type.is_some());
            if let Some(type_path) = inferred_type {
                instructions.push(Instruction::PushTypePath(type_path));
            }
            instructions.push(Instruction::TypePredicate {
                kind: *kind,
                argument_count: u8::try_from(argument_count)
                    .expect("predicate argument count was validated by the parser"),
            });
        }
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::LoadLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::LoadField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::LoadGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::Src => instructions.push(Instruction::LoadSrc),
        Expression::Usr => instructions.push(Instruction::LoadUsr),
        Expression::Caller => instructions.push(Instruction::LoadCaller),
        Expression::World => instructions.push(Instruction::LoadGlobal(
            FieldName::parse("world").expect("built-in world global name is valid"),
        )),
        Expression::GlobalNamespace => {
            return Err(compile_error("global namespace requires a field name"));
        }
        Expression::Field { receiver, name } => {
            if name.as_str() == "vars" {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::LoadDatumVars);
            } else if let Some(storage) =
                locals.receiver_static(receiver.as_ref(), name).or_else(|| {
                    matches!(receiver.as_ref(), Expression::Src)
                        .then(|| locals.global_field(name.as_str()))
                        .flatten()
                })
            {
                instructions.push(Instruction::LoadGlobal(storage.clone()));
            } else {
                let declared = expression_static_type(receiver, locals).is_some();
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(if declared {
                    Instruction::LoadDeclaredField(name.clone())
                } else {
                    Instruction::LoadField(name.clone())
                });
            }
        }
        Expression::SafeField { receiver, name } => {
            let declared = expression_static_type(receiver, locals).is_some();
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            instructions.push(if declared {
                Instruction::LoadDeclaredField(name.clone())
            } else {
                Instruction::LoadField(name.clone())
            });
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::GlobalField(name) => {
            if name.as_str() == "vars" {
                instructions.push(Instruction::LoadGlobalVars);
            } else {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
        }
        Expression::Result => instructions.push(Instruction::LoadResult),
        Expression::ArgList(_) => {
            return Err(compile_error(
                "arglist may only appear in a call or constructor argument list",
            ));
        }
        Expression::StandardBuiltin { name, arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("native builtin has more than 65535 arguments"))?;
            // `newlist(/type, ...)` is syntax sugar for constructing each
            // argument with an ordinary zero-argument `new`, then collecting
            // the resulting objects into a list.  Lower it to AllocateDatum
            // so inherited defaults, instance initializers, New(), scheduler
            // suspension, and atom registration are identical to explicit
            // construction.  A project-defined /proc/newlist still wins.
            if name == "newlist" && !procedures.contains_key(name) {
                for argument in arguments {
                    if matches!(argument, Expression::NamedArgument { .. }) {
                        return Err(compile_error("newlist does not take named arguments"));
                    }
                    emit_expression(argument, locals, instructions, procedures)?;
                    instructions.push(Instruction::AllocateDatum {
                        argument_count: 0,
                        argument_names: Vec::new(),
                    });
                }
                instructions.push(Instruction::MakeList(argument_count));
                return Ok(());
            }
            for argument in arguments {
                if let Expression::ArgList(value) = argument {
                    // A single expanded list is already the native ABI used
                    // by list-aware builtins such as min/max.
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            // DM source may deliberately replace a global procedure whose name
            // also has an engine fallback (tgstation's /proc/qdel is the
            // important case). A real project procedure wins over the native
            // fallback exactly like any other global proc declaration.
            if let Some(procedure) = procedures.get(name).copied() {
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                    argument_names: arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::NamedArgument { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                });
            } else {
                instructions.push(Instruction::StandardBuiltin {
                    name: name.clone(),
                    argument_count,
                    argument_names: arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::NamedArgument { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                });
            }
        }
        Expression::NativeSrcMethod { name, arguments } => {
            // Several BYOND engine method names are also valid project proc
            // names. Monkestation's legacy spritesheet datum declares
            // `/datum/asset/spritesheet/proc/Insert`; that project method must
            // win over `/icon.Insert`, just like project global builtins do.
            if let Some(procedure) = procedures.get(name).copied() {
                let argument_count =
                    emit_call_arguments(arguments, locals, instructions, procedures)?;
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                    argument_names: vec![None; arguments.len()],
                });
            } else {
                let argument_count = u16::try_from(arguments.len())
                    .map_err(|_| compile_error("native method has more than 65535 arguments"))?;
                for argument in arguments {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
                instructions.push(Instruction::NativeSrcMethod {
                    name: name.clone(),
                    argument_count,
                });
            }
        }
        Expression::ExternalCall {
            library,
            function,
            arguments,
        } => {
            emit_expression(library, locals, instructions, procedures)?;
            emit_expression(function, locals, instructions, procedures)?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::ExternalCall {
                argument_count: u16::try_from(arguments.len())
                    .map_err(|_| compile_error("external call has more than 65535 arguments"))?,
            });
        }
        Expression::Animate { arguments } => {
            let mut expanded_indices = Vec::new();
            for (index, (_, argument)) in arguments.iter().enumerate() {
                if let Expression::ArgList(value) = argument {
                    expanded_indices.push(to_local_index(index)?);
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            instructions.push(Instruction::Animate {
                argument_names: arguments.iter().map(|(name, _)| name.clone()).collect(),
                expanded_indices,
            });
        }
        Expression::Filter { arguments } => {
            let mut expanded_indices = Vec::new();
            for (index, (_, argument)) in arguments.iter().enumerate() {
                if let Expression::ArgList(value) = argument {
                    expanded_indices.push(to_local_index(index)?);
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            instructions.push(Instruction::MakeFilter {
                argument_names: arguments.iter().map(|(name, _)| name.clone()).collect(),
                expanded_indices,
            });
        }
        Expression::Crash(message) => {
            emit_expression(message, locals, instructions, procedures)?;
            instructions.push(Instruction::Crash);
            // Keep expression stack shape valid for unreachable continuation.
            instructions.push(Instruction::PushNull);
        }
        Expression::Sleep(delay) => {
            emit_expression(delay, locals, instructions, procedures)?;
            instructions.push(Instruction::Sleep);
        }
        Expression::Initial(reference) => match reference.as_ref() {
            Expression::Field { receiver, name } => {
                if let Some(storage) = locals.receiver_static(receiver, name) {
                    if matches!(receiver.as_ref(), Expression::TypePath(_)) {
                        // `Type::name` is DM's scope operator, not `initial()`:
                        // it reads the type's shared static slot live. The
                        // parser routes it through `Initial` only for syntax
                        // reuse.
                        instructions.push(Instruction::LoadGlobal(storage.clone()));
                    } else {
                        // Static initialization is materialized before
                        // procedures run and occupies its qualified persistent
                        // slot.
                        instructions.push(Instruction::LoadInitialGlobal(storage.clone()));
                    }
                } else {
                    emit_expression(receiver, locals, instructions, procedures)?;
                    instructions.push(Instruction::InitialField(name.clone()));
                }
            }
            Expression::Local(name) => {
                let field = locals.src_field(name).ok_or_else(|| {
                    compile_error(format!("initial target {name:?} is not an instance field"))
                })?;
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::InitialField(field.clone()));
            }
            Expression::SafeField { receiver, name } => {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let null_jump = instructions.len();
                instructions.push(Instruction::JumpIfNull(usize::MAX));
                instructions.push(Instruction::InitialField(name.clone()));
                let end = instructions.len();
                instructions[null_jump] = Instruction::JumpIfNull(end);
            }
            Expression::Index { list, index } if matches!(list.as_ref(), Expression::Field { name, .. } if name.as_str() == "vars") =>
            {
                let Expression::Field { receiver, .. } = list.as_ref() else {
                    unreachable!("vars index guard established a field receiver")
                };
                emit_expression(receiver, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::InitialDynamicField);
            }
            _ => return Err(compile_error("initial requires a field reference")),
        },
        Expression::Call {
            procedure,
            arguments,
        } => {
            let target = procedures
                .get(procedure)
                .copied()
                .ok_or_else(|| compile_error(format!("unknown procedure {procedure:?}")))?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::Call {
                procedure: target,
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
            });
        }
        Expression::Locate { arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("locate has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Locate { argument_count });
        }
        Expression::LocateIn {
            arguments,
            container,
        } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("locate has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            emit_expression(container, locals, instructions, procedures)?;
            instructions.push(Instruction::LocateIn { argument_count });
        }
        Expression::CurrentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                Some(emit_call_arguments(
                    arguments,
                    locals,
                    instructions,
                    procedures,
                )?)
            } else {
                None
            };
            instructions.push(Instruction::CallCurrent { argument_count });
        }
        Expression::ParentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                Some(emit_call_arguments(
                    arguments,
                    locals,
                    instructions,
                    procedures,
                )?)
            } else {
                None
            };
            instructions.push(Instruction::CallParent {
                procedure: procedures.get("..").copied(),
                argument_count,
            });
        }
        Expression::DynamicCall {
            target,
            procedure,
            arguments,
            null_receiver_is_global,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            let static_selector = if let Expression::Text(selector) = procedure.as_ref() {
                Some(selector.clone())
            } else {
                emit_expression(procedure, locals, instructions, procedures)?;
                None
            };
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic {
                static_selector,
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
                null_receiver_is_global: *null_receiver_is_global,
            });
        }
        Expression::SafeDynamicCall {
            target,
            procedure,
            arguments,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            let static_selector = if let Expression::Text(selector) = procedure.as_ref() {
                Some(selector.clone())
            } else {
                emit_expression(procedure, locals, instructions, procedures)?;
                None
            };
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic {
                static_selector,
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
                null_receiver_is_global: false,
            });
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::List(entries) => {
            let mut kinds = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Positional);
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        emit_associative_list_key(key, locals, instructions, procedures)?;
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Associative);
                    }
                }
            }
            instructions.push(Instruction::MakeListEntries(kinds));
        }
        Expression::AssociativeList(entries) => {
            let mut kinds = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Positional);
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        emit_associative_list_key(key, locals, instructions, procedures)?;
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Associative);
                    }
                }
            }
            instructions.push(Instruction::MakeAssociativeListEntries(kinds));
        }
        Expression::Index { list, index } => {
            if let Expression::Field { receiver, name } = list.as_ref()
                && name.as_str() == "vars"
            {
                emit_expression(receiver, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::LoadDynamicField);
            } else if let Expression::Local(name) = list.as_ref()
                && let Some(slot) = locals.get(name)
            {
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::IndexLocalList(slot));
            } else {
                emit_expression(list, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::IndexList);
            }
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::IndexList);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::Unary { operator, operand } => {
            if operator == "&"
                && let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::AddressLocal(slot));
                return Ok(());
            }
            if operator == "*"
                && let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::LoadLocalRaw(slot));
                instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
                instructions.push(Instruction::IndexList);
                return Ok(());
            }
            emit_expression(operand, locals, instructions, procedures)?;
            match operator.as_str() {
                "+" => {}
                "-" => instructions.push(Instruction::Negate),
                "!" => instructions.push(Instruction::Not),
                "~" => instructions.push(Instruction::BitNot),
                "&" => instructions.push(Instruction::MakeList(1)),
                "*" => {
                    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
                    instructions.push(Instruction::IndexList);
                }
                _ => {
                    return Err(compile_error(format!(
                        "unsupported unary operator {operator}"
                    )));
                }
            }
        }
        Expression::Mutation {
            target,
            delta,
            prefix,
        } => emit_mutation_expression(target, *delta, *prefix, locals, instructions, procedures)?,
        Expression::Binary {
            operator,
            left,
            right,
        } => {
            if operator == "&&" {
                emit_expression(left, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                instructions.push(Instruction::Pop);
                emit_expression(right, locals, instructions, procedures)?;
                let end = instructions.len();
                patch_jump(instructions, false_jump, end)?;
            } else if operator == "||" {
                emit_expression(left, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                let end_jump = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let false_target = instructions.len();
                patch_jump(instructions, false_jump, false_target)?;
                instructions.push(Instruction::Pop);
                emit_expression(right, locals, instructions, procedures)?;
                let end = instructions.len();
                patch_jump(instructions, end_jump, end)?;
            } else {
                emit_expression(left, locals, instructions, procedures)?;
                emit_expression(right, locals, instructions, procedures)?;
                instructions.push(match operator.as_str() {
                    "+" => Instruction::Add,
                    "-" => Instruction::Subtract,
                    "*" => Instruction::Multiply,
                    "**" => Instruction::Power,
                    "/" => Instruction::Divide,
                    "%" => Instruction::Remainder,
                    "%%" => Instruction::FractionalRemainder,
                    "&" => Instruction::BitAnd,
                    "|" => Instruction::BitOr,
                    "^" => Instruction::BitXor,
                    "<<" => Instruction::ShiftLeft,
                    ">>" => Instruction::ShiftRight,
                    "==" => Instruction::Equal,
                    "!=" | "<>" => Instruction::NotEqual,
                    "~=" => Instruction::Equivalent,
                    "~!" => Instruction::NotEquivalent,
                    "<=>" => Instruction::Compare,
                    "in" => Instruction::Contains,
                    "<" => Instruction::Less,
                    "<=" => Instruction::LessEqual,
                    ">" => Instruction::Greater,
                    ">=" => Instruction::GreaterEqual,
                    _ => {
                        return Err(compile_error(format!(
                            "unsupported binary operator {operator}"
                        )));
                    }
                });
            }
        }
        Expression::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            emit_expression(condition, locals, instructions, procedures)?;
            let false_jump = instructions.len();
            instructions.push(Instruction::JumpIfFalse(usize::MAX));
            emit_expression(when_true, locals, instructions, procedures)?;
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let false_target = instructions.len();
            patch_jump(instructions, false_jump, false_target)?;
            emit_expression(when_false, locals, instructions, procedures)?;
            let end_target = instructions.len();
            patch_jump(instructions, end_jump, end_target)?;
        }
        Expression::LogicalOrAssignment { target, value } => {
            if !matches!(value.as_ref(), Expression::List(entries) if entries.is_empty())
                || !emit_logical_or_empty_list_assignment(target, locals, instructions, procedures)?
            {
                // Keep every non-empty RHS on the general logical-assignment
                // lowering. The superinstructions below are deliberately
                // exact to the overwhelmingly common empty-list constructor.
                emit_expression(target, locals, instructions, procedures)?;
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                emit_expression(target, locals, instructions, procedures)?;
                let end_jump = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let false_target = instructions.len();
                patch_jump(instructions, false_jump, false_target)?;
                emit_assignment_expression(target, "=", value, locals, instructions, procedures)?;
                let end_target = instructions.len();
                patch_jump(instructions, end_jump, end_target)?;
            }
        }
        Expression::Assignment {
            target,
            operator,
            value,
        } => emit_assignment_expression(target, operator, value, locals, instructions, procedures)?,
    }
    Ok(())
}

fn emit_logical_or_empty_list_assignment(
    target: &Expression,
    locals: &LocalTable<'_>,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<bool, CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::LogicalOrEmptyListLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::LogicalOrEmptyListField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::LogicalOrEmptyListGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) if name.as_str() != "vars" => {
            instructions.push(Instruction::LogicalOrEmptyListGlobal(name.clone()));
        }
        Expression::Field { receiver, name } if name.as_str() != "vars" => {
            if let Some(storage) = locals.receiver_static(receiver.as_ref(), name).or_else(|| {
                matches!(receiver.as_ref(), Expression::Src)
                    .then(|| locals.global_field(name.as_str()))
                    .flatten()
            }) {
                instructions.push(Instruction::LogicalOrEmptyListGlobal(storage.clone()));
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::LogicalOrEmptyListField(name.clone()));
            }
        }
        Expression::Initial(reference)
            if matches!(
                reference.as_ref(),
                Expression::Field { receiver, name }
                    if name.as_str() != "vars"
                        && locals.receiver_static(receiver, name).is_some()
            ) =>
        {
            let Expression::Field { receiver, name } = reference.as_ref() else {
                unreachable!("guard matched a field reference")
            };
            let storage = locals
                .receiver_static(receiver, name)
                .expect("guard resolved the static slot")
                .clone();
            instructions.push(Instruction::LogicalOrEmptyListGlobal(storage));
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::LogicalOrEmptyListIndex);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn emit_mutation_expression(
    target: &Expression,
    delta: i8,
    prefix: bool,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::MutateLocal {
                    slot,
                    delta,
                    prefix,
                });
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::MutateField {
                    name: field.clone(),
                    delta,
                    prefix,
                });
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::MutateGlobal {
                    name: global.clone(),
                    delta,
                    prefix,
                });
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) => instructions.push(Instruction::MutateGlobal {
            name: name.clone(),
            delta,
            prefix,
        }),
        Expression::Field { receiver, name } => {
            if let Some(storage) = locals.receiver_static(receiver.as_ref(), name) {
                instructions.push(Instruction::MutateGlobal {
                    name: storage.clone(),
                    delta,
                    prefix,
                });
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::MutateField {
                    name: name.clone(),
                    delta,
                    prefix,
                });
            }
        }
        Expression::SafeField { receiver, name } => {
            if let Some(storage) = locals.receiver_static(receiver.as_ref(), name) {
                instructions.push(Instruction::MutateGlobal {
                    name: storage.clone(),
                    delta,
                    prefix,
                });
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let null_jump = instructions.len();
                instructions.push(Instruction::JumpIfNull(usize::MAX));
                instructions.push(Instruction::MutateField {
                    name: name.clone(),
                    delta,
                    prefix,
                });
                let end = instructions.len();
                instructions[null_jump] = Instruction::JumpIfNull(end);
            }
        }
        // `Type::name++` — the scope operator parses through `Initial`; mutate
        // the resolved shared static slot in place.
        Expression::Initial(reference)
            if matches!(
                reference.as_ref(),
                Expression::Field { receiver, name }
                    if locals.receiver_static(receiver, name).is_some()
            ) =>
        {
            let Expression::Field { receiver, name } = reference.as_ref() else {
                unreachable!("guard matched a field reference")
            };
            let storage = locals
                .receiver_static(receiver, name)
                .expect("guard resolved the static slot")
                .clone();
            instructions.push(Instruction::MutateGlobal {
                name: storage,
                delta,
                prefix,
            });
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::MutateListIndex { delta, prefix });
        }
        Expression::Result => instructions.push(Instruction::MutateResult { delta, prefix }),
        _ => return Err(compile_error("increment/decrement target is not writable")),
    }
    Ok(())
}

/// Emits an assignment to a qualified static slot in expression position,
/// leaving the assigned value on the stack. Mirrors the `GlobalField` arm of
/// [`emit_assignment_expression`], including compound-operator handling.
fn emit_qualified_static_assignment(
    storage: FieldName,
    operator: &str,
    value: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    if operator != "=" {
        instructions.push(Instruction::LoadGlobal(storage.clone()));
    }
    emit_expression(value, locals, instructions, procedures)?;
    if operator != "=" {
        instructions.push(compound_instruction(operator)?);
    }
    instructions.push(Instruction::Duplicate);
    instructions.push(Instruction::StoreGlobal(storage));
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn emit_assignment_expression(
    target: &Expression,
    operator: &str,
    value: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Result => {
            if operator != "=" {
                instructions.push(Instruction::LoadResult);
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreResult);
        }
        Expression::Usr => {
            if operator != "=" {
                return Err(compile_error("usr only supports direct assignment"));
            }
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreUsr);
        }
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                if operator != "=" {
                    instructions.push(Instruction::LoadLocal(slot));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                if operator != "=" {
                    instructions.push(Instruction::Duplicate);
                    instructions.push(Instruction::LoadField(field.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreFieldKeep(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                if operator != "=" {
                    instructions.push(Instruction::LoadGlobal(global.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) => {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreGlobal(name.clone()));
        }
        Expression::Src => {
            if operator != "=" {
                return Err(compile_error("src only supports direct assignment"));
            }
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreSrc);
        }
        Expression::Field { receiver, name } => {
            if let Some(storage) = locals.receiver_static(receiver, name) {
                emit_qualified_static_assignment(
                    storage.clone(),
                    operator,
                    value,
                    locals,
                    instructions,
                    procedures,
                )?;
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(Instruction::Duplicate);
                    instructions.push(Instruction::LoadField(name.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreFieldKeep(name.clone()));
            }
        }
        // DM's `Type::name` scope operator parses through `Initial`; a writable
        // target here is a literal type's shared static, resolved by
        // dm-semantics to a qualified `__dm_static_` slot.
        Expression::Initial(reference)
            if matches!(
                reference.as_ref(),
                Expression::Field { receiver, name }
                    if locals.receiver_static(receiver, name).is_some()
            ) =>
        {
            let Expression::Field { receiver, name } = reference.as_ref() else {
                unreachable!("guard matched a field reference")
            };
            let storage = locals
                .receiver_static(receiver, name)
                .expect("guard resolved the static slot")
                .clone();
            emit_qualified_static_assignment(
                storage,
                operator,
                value,
                locals,
                instructions,
                procedures,
            )?;
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreFieldKeep(name.clone()));
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::Index { list, index } => {
            if operator == "=" {
                emit_expression(value, locals, instructions, procedures)?;
                emit_expression(list, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::PrepareRhsFirstIndexAssignment);
                instructions.push(Instruction::SetListIndexKeep);
            } else {
                emit_expression(list, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::CompoundListIndexKeep(
                    compound_list_index_operator(operator)?,
                ));
            }
        }
        Expression::SafeIndex { list, index } => {
            if operator != "=" {
                return Err(compile_error(
                    "compound null-conditional list assignment is not supported as an expression",
                ));
            }
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::SetListIndexKeep);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        _ => return Err(compile_error("assignment target is not writable")),
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(crate) fn bind_initializer_expression(
    expression: &mut Expression,
    bindings: &BTreeMap<String, InitializerBinding>,
) -> Result<(), CompileError> {
    match expression {
        Expression::World => {}
        Expression::Local(name) => {
            let binding = bindings
                .get(name)
                .ok_or_else(|| compile_error(format!("unresolved initializer name {name:?}")))?;
            *expression = match binding {
                InitializerBinding::Global(field) => Expression::GlobalField(field.clone()),
                InitializerBinding::SrcField(field) => Expression::Field {
                    receiver: Box::new(Expression::Src),
                    name: field.clone(),
                },
            };
        }
        Expression::Field { receiver, name } | Expression::SafeField { receiver, name } => {
            if let Expression::Local(receiver_name) = receiver.as_ref()
                && let Some(InitializerBinding::Global(storage)) =
                    bindings.get(&format!("{receiver_name}.{}", name.as_str()))
            {
                *expression = Expression::GlobalField(storage.clone());
            } else {
                bind_initializer_expression(receiver, bindings)?;
            }
        }
        Expression::NamedArgument { value, .. } => {
            bind_initializer_expression(value, bindings)?;
        }
        Expression::Call { arguments, .. }
        | Expression::StandardBuiltin { arguments, .. }
        | Expression::NativeSrcMethod { arguments, .. }
        | Expression::Regex { arguments }
        | Expression::MutableAppearance { arguments }
        | Expression::Matrix { arguments }
        | Expression::Vector { arguments }
        | Expression::ReplaceText { arguments, .. }
        | Expression::CopyText { arguments, .. }
        | Expression::Block { arguments }
        | Expression::Rand { arguments }
        | Expression::Roll { arguments }
        | Expression::Round { arguments }
        | Expression::Range { arguments }
        | Expression::TypePredicate { arguments, .. }
        | Expression::Locate { arguments } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::ExternalCall {
            library,
            function,
            arguments,
        } => {
            bind_initializer_expression(library, bindings)?;
            bind_initializer_expression(function, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Animate { arguments } => {
            for (_, argument) in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Filter { arguments } => {
            for (_, argument) in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::TypesOf { arguments } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::HasCall { receiver, selector } => {
            bind_initializer_expression(receiver, bindings)?;
            bind_initializer_expression(selector, bindings)?;
        }
        Expression::Length { value }
        | Expression::Ref { value }
        | Expression::Initial(value)
        | Expression::Sleep(value)
        | Expression::Crash(value) => {
            bind_initializer_expression(value, bindings)?;
        }
        Expression::ArgList(value) => bind_initializer_expression(value, bindings)?,
        Expression::GetStep { source, direction } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(direction, bindings)?;
        }
        Expression::GetStepTowards { source, target } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(target, bindings)?;
        }
        Expression::Prob(chance) => bind_initializer_expression(chance, bindings)?,
        Expression::Pick { entries } => {
            for (weight, candidate) in entries {
                if let Some(weight) = weight {
                    bind_initializer_expression(weight, bindings)?;
                }
                bind_initializer_expression(candidate, bindings)?;
            }
        }
        Expression::New {
            type_path,
            arguments,
            overrides,
        } => {
            if let Some(type_path) = type_path {
                bind_initializer_expression(type_path, bindings)?;
            }
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
            for (_, value) in overrides {
                bind_initializer_expression(value, bindings)?;
            }
        }
        Expression::ModifiedTypePath { overrides, .. } => {
            for (_, value) in overrides {
                bind_initializer_expression(value, bindings)?;
            }
        }
        Expression::LocateIn {
            arguments,
            container,
        } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
            bind_initializer_expression(container, bindings)?;
        }
        Expression::DynamicCall {
            target,
            procedure,
            arguments,
            ..
        }
        | Expression::SafeDynamicCall {
            target,
            procedure,
            arguments,
        } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(procedure, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::List(entries) | Expression::AssociativeList(entries) => {
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        bind_initializer_expression(value, bindings)?;
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        // A bare key in `list(name = value)` is named-argument
                        // syntax and therefore the text "name", even if an
                        // initializer binding with that spelling exists.
                        let bare_text_key = matches!(key, Expression::Local(_));
                        if !bare_text_key {
                            bind_initializer_expression(key, bindings)?;
                        }
                        bind_initializer_expression(value, bindings)?;
                    }
                }
            }
        }
        Expression::Index { list, index } | Expression::SafeIndex { list, index } => {
            bind_initializer_expression(list, bindings)?;
            bind_initializer_expression(index, bindings)?;
        }
        Expression::Unary { operand, .. }
        | Expression::Mutation {
            target: operand, ..
        } => {
            bind_initializer_expression(operand, bindings)?;
        }
        Expression::Binary { left, right, .. } => {
            bind_initializer_expression(left, bindings)?;
            bind_initializer_expression(right, bindings)?;
        }
        Expression::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            bind_initializer_expression(condition, bindings)?;
            bind_initializer_expression(when_true, bindings)?;
            bind_initializer_expression(when_false, bindings)?;
        }
        Expression::LogicalOrAssignment { target, value }
        | Expression::Assignment { target, value, .. } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(value, bindings)?;
        }
        Expression::CurrentCall { .. }
        | Expression::ParentCall { .. }
        | Expression::Result
        | Expression::Caller => {
            return Err(compile_error(
                "current-procedure state is unavailable in a variable initializer",
            ));
        }
        Expression::Null
        | Expression::Number(_)
        | Expression::Text(_)
        | Expression::File(_)
        | Expression::TypePath(_)
        | Expression::Src
        | Expression::Usr
        | Expression::GlobalNamespace
        | Expression::GlobalField(_) => {}
    }
    Ok(())
}
