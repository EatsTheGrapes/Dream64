use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::CompilerDatabase;
use dm_core::SourceSpan;
use dm_object_tree::NodeKind;
use dm_syntax::DefinitionKind;

static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

struct TestProject {
    root: PathBuf,
}

impl TestProject {
    fn new() -> Self {
        let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dream64-mapping-helper-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("test project should be created");
        Self { root }
    }

    fn write(&self, path: &str, contents: &str) {
        fs::write(self.root.join(path), contents).expect("fixture file should be written");
    }
}

impl Drop for TestProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn mapping_helper_expansion_enters_the_tree_at_its_invocation_span() {
    let fixture = TestProject::new();
    fixture.write(
        "world.dme",
        "#include \"defines.dm\"\n#include \"types.dm\"\n",
    );
    fixture.write(
        "defines.dm",
        concat!(
            "#define HELPERS(path, offset) ##path/directional/north {\\\n",
            "\tdir = 1; \\\n",
            "\tpixel_y = offset; \\\n",
            "} \\\n",
            "##path/directional/east {\\\n",
            "\tdir = 4; \\\n",
            "\tpixel_x = offset; \\\n",
            "}\n",
        ),
    );
    let invocation = "HELPERS(/obj/light, 7)";
    fixture.write("types.dm", &format!("{invocation}\n"));

    let compilation = CompilerDatabase::new()
        .compile(fixture.root.join("world.dme"))
        .expect("mapping helper project should compile");
    assert_eq!(compilation.stats().errors, 0);
    let source_file = compilation
        .project()
        .files
        .iter()
        .find(|file| file.relative_path == Path::new("types.dm"))
        .expect("invocation source should be discovered");
    let expanded = source_file
        .compiler_text()
        .expect("expanded source should remain UTF-8");
    assert!(expanded.contains("/obj/light/directional/north"));
    assert!(expanded.contains("/obj/light/directional/east"));

    let syntax = compilation
        .syntax(source_file.id)
        .expect("expanded mapping helpers should parse");
    assert_eq!(syntax.definitions.len(), 6);
    for definition in &syntax.definitions {
        assert_eq!(
            compilation.original_span(source_file.id, definition.span),
            Some(SourceSpan::new(0, invocation.len()))
        );
    }
    let generated_types: Vec<_> = syntax
        .definitions
        .iter()
        .filter(|definition| definition.kind == DefinitionKind::Type)
        .collect();
    assert_eq!(generated_types.len(), 2);
    for (path, kind) in [
        ("/obj/light/directional/north", NodeKind::Type),
        ("/obj/light/directional/north/var/dir", NodeKind::Variable),
        (
            "/obj/light/directional/north/var/pixel_y",
            NodeKind::Variable,
        ),
        ("/obj/light/directional/east", NodeKind::Type),
        ("/obj/light/directional/east/var/dir", NodeKind::Variable),
        (
            "/obj/light/directional/east/var/pixel_x",
            NodeKind::Variable,
        ),
    ] {
        let node = compilation
            .code_tree()
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == path)
            .expect("generated mapping helper should enter the object tree");
        assert_eq!(node.kind, kind);
    }
}
