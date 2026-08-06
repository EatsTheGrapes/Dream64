use dm_syntax::{DefinitionKind, parse};

#[test]
fn expands_adjacent_braced_type_declarations_and_members() {
    let source = concat!(
        "/obj/light/directional/north { dir = 1; pixel_y = 2; } ",
        "/obj/light/directional/south { dir = 2; pixel_y = -2; }\n",
    );
    let syntax = parse(source).expect("braced declarations should parse");
    let indexed: Vec<_> = syntax
        .definitions
        .iter()
        .map(|definition| {
            (
                definition.path.to_string(),
                definition.kind,
                definition.parent,
            )
        })
        .collect();

    assert_eq!(
        indexed,
        [
            (
                "/obj/light/directional/north".to_owned(),
                DefinitionKind::Type,
                None,
            ),
            (
                "/obj/light/directional/north/var/dir".to_owned(),
                DefinitionKind::VariableOverride,
                Some(0),
            ),
            (
                "/obj/light/directional/north/var/pixel_y".to_owned(),
                DefinitionKind::VariableOverride,
                Some(0),
            ),
            (
                "/obj/light/directional/south".to_owned(),
                DefinitionKind::Type,
                None,
            ),
            (
                "/obj/light/directional/south/var/dir".to_owned(),
                DefinitionKind::VariableOverride,
                Some(3),
            ),
            (
                "/obj/light/directional/south/var/pixel_y".to_owned(),
                DefinitionKind::VariableOverride,
                Some(3),
            ),
        ]
    );
}

#[test]
fn splits_newline_separated_members_in_a_braced_type_body() {
    let syntax = parse("/datum/example {\nfirst = 1\nsecond = 2\n}\n")
        .expect("multiline braced body should parse");

    assert_eq!(syntax.definitions.len(), 3);
    assert_eq!(
        syntax.definitions[1].path.to_string(),
        "/datum/example/var/first"
    );
    assert_eq!(
        syntax.definitions[2].path.to_string(),
        "/datum/example/var/second"
    );
}

#[test]
fn retains_semicolon_prefix_and_trailing_owner_around_a_braced_proc() {
    let source = concat!(
        "var/global/datum/controller/subsystem/air/SSair; ",
        "/datum/controller/subsystem/air/New() { ss_id = \"air\"; } ",
        "/datum/controller/subsystem/air\n",
        "\tvar/list/currentrun = list()\n",
    );
    let syntax = parse(source).expect("macro-like declaration sequence should parse");
    let indexed: Vec<_> = syntax
        .definitions
        .iter()
        .map(|definition| (definition.path.to_string(), definition.kind))
        .collect();

    assert_eq!(
        indexed,
        [
            ("/var/SSair".to_owned(), DefinitionKind::Variable),
            (
                "/datum/controller/subsystem/air/proc/New".to_owned(),
                DefinitionKind::ProcedureOverride,
            ),
            (
                "/datum/controller/subsystem/air".to_owned(),
                DefinitionKind::Type,
            ),
            (
                "/datum/controller/subsystem/air/var/currentrun".to_owned(),
                DefinitionKind::Variable,
            ),
        ]
    );
    assert_eq!(syntax.definitions[1].body.len(), 1);
    assert_eq!(syntax.definitions[3].parent, Some(2));
}
