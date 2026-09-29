//! Single shared mutex for tests that mutate process-global environment.
//!
//! The process environment is global to the test binary, so any two tests that
//! call `std::env::set_var` / `remove_var` race with each other regardless of
//! which module they live in. Serialising them requires exactly one lock: a
//! per-module mutex excludes nothing, because a `session` test holding the
//! `session` mutex still ran concurrently with an `integration` test holding a
//! different mutex.
//!
//! Historically this crate kept one `static LOCK` per test module
//! (`integration_env_lock`, `test_config_env_lock`, and seven private
//! `env_lock`/`remote_env_lock` helpers). Those are all retained as thin
//! wrappers over [`lock`] so existing call sites keep working while sharing a
//! single mutex.
//!
//! # Re-entrancy
//!
//! [`lock`] hands out a [`std::sync::Mutex`], which is **not** reentrant. The
//! lock must therefore never be acquired twice on one thread. Module-local
//! helpers that mutate the environment while their caller already holds the
//! lock are deliberately left without a nested acquisition:
//!
//! * `src/detect/manifest_update.rs` `with_state_dir`
//! * `src/update.rs` `set_test_config_home`
//! * `src/app/mod.rs` `restore_xdg_state_home`
//!
//! Adding an acquisition inside any of those would self-deadlock, because their
//! callers already hold [`lock`].
//!
//! # Poisoning
//!
//! [`lock`] recovers from poisoning instead of propagating it. A single test
//! that panics while holding the guard would otherwise poison the mutex and
//! turn every later env-mutating test into a `PoisonError` panic, masking one
//! real failure behind dozens of cascading ones. The environment mutations a
//! test performs are restored by that test's own `Drop` guards, so continuing
//! after a poison is safe here.

use std::sync::{Mutex, MutexGuard, OnceLock};

static LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// The one mutex every env-mutating test must hold.
///
/// Acquiring it returns a guard that is released on drop, at which point the
/// calling test's `Drop` implementations restore the variables they changed.
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A guard for callers that need the raw `&'static Mutex`, such as code that
/// locks before doing work that must stay outside the guard's scope.
pub(crate) fn raw() -> &'static Mutex<()> {
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the cascade that made this crate's suite look
    /// flaky: one panicking test used to poison a shared mutex and fail every
    /// later env-mutating test with `PoisonError`.
    #[test]
    fn lock_recovers_from_poisoning() {
        let path = std::thread::spawn(|| {
            let _guard = lock();
            panic!("deliberate panic to poison the shared env lock");
        });
        assert!(path.join().is_err(), "child thread should have panicked");

        // A poisoned mutex must still hand out a usable guard.
        let guard = lock();
        drop(guard);
    }

    /// The whole point of this module: one mutex, not one per module.
    /// `integration_env_lock()` returns a guard rather than the mutex, so
    /// `test_config_env_lock()` is the accessor a pointer identity check can
    /// use; the integration accessor shares the mutex by construction.
    #[test]
    fn env_lock_accessors_share_one_mutex() {
        let shared = raw() as *const Mutex<()>;
        let via_config = crate::config::test_config_env_lock() as *const Mutex<()>;
        assert!(std::ptr::eq(shared, via_config));
    }
}
