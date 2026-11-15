// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Browser glue for the WebAssembly build.

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

fn window() -> web_sys::Window {
    web_sys::window().expect("no window")
}

fn js_err(e: wasm_bindgen::JsValue) -> String {
    e.as_string().or_else(|| js_sys::JSON::stringify(&e).ok().and_then(|s| s.as_string())).unwrap_or_default()
}

/// Download a file with `fetch`.
pub async fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let resp = JsFuture::from(window().fetch_with_str(url))
        .await
        .map_err(|e| format!("{url}: {}", js_err(e)))?;
    let resp: web_sys::Response = resp.dyn_into().map_err(js_err)?;
    if !resp.ok() {
        return Err(format!("{url}: HTTP {} {}", resp.status(), resp.status_text()));
    }
    let buf = JsFuture::from(resp.array_buffer().map_err(js_err)?).await.map_err(js_err)?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

/// Give the browser a chance to render and handle input during long work.
pub async fn yield_now() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let _ = window().set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
    });
    let _ = JsFuture::from(promise).await;
}

pub fn storage_get(key: &str) -> Option<String> {
    window().local_storage().ok().flatten()?.get_item(key).ok().flatten()
}

pub fn storage_set(key: &str, value: &str) {
    if let Some(s) = window().local_storage().ok().flatten() {
        let _ = s.set_item(key, value);
    }
}

pub fn set_title(title: &str) {
    if let Some(d) = window().document() {
        d.set_title(title);
    }
}

pub fn is_fullscreen() -> bool {
    window().document().is_some_and(|d| d.fullscreen_element().is_some())
}

/// Enter or leave fullscreen. Browsers only allow this shortly after a user
/// action (click or key press), which is how the app calls it.
pub fn set_fullscreen(on: bool) {
    let Some(doc) = window().document() else { return };
    if on {
        if let Some(el) = doc.document_element() {
            let _ = el.request_fullscreen();
        }
    } else if doc.fullscreen_element().is_some() {
        doc.exit_fullscreen();
    }
}

/// Query parameters: `?pano=<url>&yaw=&pitch=&hfov=&fs=1`.
pub fn url_params() -> Vec<(String, String)> {
    let Ok(search) = window().location().search() else { return Vec::new() };
    let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search) else { return Vec::new() };
    ["pano", "yaw", "pitch", "hfov", "fs"]
        .iter()
        .filter_map(|k| params.get(k).map(|v| (k.to_string(), v)))
        .collect()
}

pub fn canvas() -> Option<web_sys::HtmlCanvasElement> {
    window().document()?.get_element_by_id("app")?.dyn_into().ok()
}

/// Remove the startup screen (icon) once the app runs.
pub fn hide_loading_screen() {
    if let Some(el) = window().document().and_then(|d| d.get_element_by_id("loading")) {
        el.remove();
    }
}

pub fn show_startup_error(message: &str) {
    if let Some(el) = window().document().and_then(|d| d.get_element_by_id("loading-text")) {
        el.set_text_content(Some(message));
    }
}
