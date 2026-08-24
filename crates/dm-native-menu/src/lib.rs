//! Audited platform boundary for attaching Dream64's native application menu.

#[cfg(windows)]
pub use muda;

/// Attaches a native menu to a live Win32 window handle.
///
/// The caller obtains `hwnd` from winit and must keep both the window and menu
/// alive for the duration of the attachment.
#[cfg(windows)]
pub fn install_for_hwnd(menu: &muda::Menu, hwnd: isize) -> Result<(), String> {
    // SAFETY: this crate is the single audited FFI boundary. The public safe
    // wrapper is only called with a non-zero HWND obtained from winit's live
    // Win32 window handle, and the owning client retains the Menu until exit.
    unsafe { menu.init_for_hwnd(hwnd) }.map_err(|error| error.to_string())
}
