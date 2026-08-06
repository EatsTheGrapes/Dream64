use dm_map::{MapValueKind, parse};

#[test]
fn retains_structured_map_values_without_evaluating_them() {
    let source = concat!(
        "\"a\" = (/obj/item{\n",
        "name = \"semi; \\\"quoted\\\"\";\n",
        "icon = 'icons/obj/items.dmi';\n",
        "allowed = list(/obj/item, \"x;y\" = /datum/test);\n",
        "target = /datum/example;\n",
        "dir = NORTH;\n",
        "count = -2.5;\n",
        "optional = null;\n",
        "computed = 1 + 2\n",
        "}, /turf, /area)\n",
        "(1,1,1) = {\"\na\n\"}\n",
    );
    let map = parse(source).expect("lossless value fixture should parse");
    let atom = &map.keys["a"].atoms[0];
    let assignments = &atom.variable_assignments;

    assert_eq!(assignments.len(), 8);
    assert_eq!(assignments[0].value.kind, MapValueKind::Text);
    assert_eq!(assignments[0].value.raw, "\"semi; \\\"quoted\\\"\"");
    assert_eq!(assignments[1].value.kind, MapValueKind::Resource);
    assert_eq!(assignments[2].value.kind, MapValueKind::List);
    assert_eq!(assignments[3].value.kind, MapValueKind::Path);
    assert_eq!(assignments[4].value.kind, MapValueKind::Identifier);
    assert_eq!(assignments[5].value.kind, MapValueKind::Number);
    assert_eq!(assignments[6].value.kind, MapValueKind::Null);
    assert_eq!(assignments[7].value.kind, MapValueKind::Expression);
    for assignment in assignments {
        assert_eq!(
            &source[assignment.span.start..assignment.span.end],
            assignment.raw
        );
        assert_eq!(
            &source[assignment.value.span.start..assignment.value.span.end],
            assignment.value.raw
        );
    }
    assert!(
        atom.variables
            .as_deref()
            .is_some_and(|raw| raw.contains("list(/obj/item, \"x;y\" = /datum/test)"))
    );
}

#[test]
fn diagnoses_unterminated_nested_map_values_at_the_opening_delimiter() {
    let source = "\"a\" = (/obj{x = list(1}, /turf, /area)\n(1,1,1) = {\"\na\n\"}\n";
    let error = parse(source).expect_err("unterminated list should fail");

    assert!(error.message.contains("unterminated '('"));
    assert_eq!(
        &source[error.span.start..error.span.end],
        "(",
        "diagnostic should identify the opening delimiter"
    );
}

#[test]
fn ignores_braces_inside_variable_block_comments() {
    let source = "\"a\" = (/obj{x = 1 /* } is inert */}, /turf, /area)\n(1,1,1) = {\"\na\n\"}\n";
    let map = parse(source).expect("comment brace should not close the initializer");

    assert_eq!(map.keys["a"].atoms[0].variable_assignments.len(), 1);
}
