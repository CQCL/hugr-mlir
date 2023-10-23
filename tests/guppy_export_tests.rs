
use rstest::{rstest,fixture};

use hugr_mlir::Result;

#[fixture]
pub fn test_context() -> melior::Context {
    let dr = melior::dialect::DialectRegistry::new();
    let hugr_dh = hugr_mlir::mlir::hugr::get_hugr_dialect_handle();
    hugr_dh.insert_dialect(&dr);
    let ctx = melior::Context::new();
    ctx.append_dialect_registry(&dr);
    ctx.load_all_available_dialects();
    ctx
}

#[rstest]
fn test_guppy_exports(test_context: melior::Context) -> Result<()> {
    insta::glob!("guppy-exports", "*.json", |path| {
        use std::fs;
        let src = fs::read(path).unwrap();
        let ul = melior::ir::Location::new(&test_context, path.to_str().unwrap(), 0, 0);
        let m = hugr_mlir::translate::translate_hugr_to_mlir(&src, ul, |bytes| serde_json::from_slice::<hugr::Hugr>(bytes)).unwrap();
        let o = m.as_operation();
        println!("{o}");
        assert!(m.as_operation().verify());
        insta::assert_display_snapshot!(m.as_operation());
    });
    Ok(())
}
