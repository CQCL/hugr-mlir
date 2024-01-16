use rstest::{fixture, rstest};
use std::fs;

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
    use hugr_mlir::hugr_to_mlir::hugr_to_mlir;
    unsafe {
        hugr_mlir::mlir::hugr::ffi::mlirHugrRegisterOptPipelines();
    }

    // unsafe { mlir_sys::mlirEnableGlobalDebug(true); }
    insta::glob!("guppy-exports", "*.json", |path| {
        let mut settings = insta::Settings::clone_current();
        settings.set_description(path.to_string_lossy());
        settings.bind(|| {
            println!("{:?}", path);

            let bytes = fs::read(path).unwrap();
            let ul = melior::ir::Location::new(&test_context, path.to_str().unwrap(), 0, 0);
            let hugr = serde_json::from_slice::<hugr::Hugr>(&bytes).unwrap();
            let mut mlir_mod = hugr_to_mlir(ul, &hugr).unwrap();
            assert!(mlir_mod.as_operation().verify());
            insta::assert_display_snapshot!(mlir_mod.as_operation());

            let pm = melior::pass::PassManager::new(&test_context);
            melior::utility::parse_pass_pipeline(
                pm.as_operation_pass_manager(),
                "builtin.module(lower-hugr)",
            )
            .unwrap();
            pm.enable_verifier(true);
            match pm.run(&mut mlir_mod) {
                Err(_) => {
                    println!("{}", mlir_mod.as_operation());
                    assert!(false);
                }
                _ => (),
            }
            insta::assert_display_snapshot!(mlir_mod.as_operation());
        });
    });
    Ok(())
}
