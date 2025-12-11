use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use tensor_engine::nn::VisionTransformer;
use tensor_engine::nn::MultimodalLLM;
use tensor_engine::tensor::Tensor;

#[cfg(feature = "server")]
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
#[cfg(feature = "server")]
use serde::Deserialize;

#[cfg(feature = "server")]
#[derive(Deserialize)]
struct GenerateRequest {
    image_b64: String,
    max_len: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

#[cfg(feature = "server")]
#[post("/generate")]
async fn generate(req: web::Json<GenerateRequest>) -> impl Responder {
    // Decode base64 image into raw bytes and save/convert via image crate in-memory
    let decoded = match STANDARD.decode(&req.image_b64) {
        Ok(d) => d,
        Err(e) => return HttpResponse::BadRequest().body(format!("Invalid base64 image: {}", e)),
    };
    // Try loading image via image crate
    let img = match image::load_from_memory(&decoded) {
        Ok(i) => i.to_rgb8(),
        Err(e) => return HttpResponse::BadRequest().body(format!("Failed to decode image bytes: {}", e)),
    };
    // Convert to Tensor expected shape [1,3,H,W]
    let (w, h) = img.dimensions();
    let mut data: Vec<f32> = Vec::with_capacity((w * h * 3) as usize);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data.push(img.get_pixel(x, y)[c] as f32 / 255.0);
            }
        }
    }
    let tensor = Tensor::new(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 3, h as usize, w as usize]), data).unwrap().into_dyn(), false);

    // Build a tiny model for demonstration (in real service you'd load weights once and reuse)
    let vit = VisionTransformer::new(3, 8, 32, 64, 4, 2, 512);
    let model = MultimodalLLM::new(vit, 512, 32, 64, 4, 2);

    let max_len = req.max_len.unwrap_or(16);
    let temp = req.temperature.unwrap_or(1.0);
    match model.generate(&tensor, None, max_len, temp, req.top_k, req.top_p, 1) {
        Ok(seq) => HttpResponse::Ok().json(seq),
        Err(e) => HttpResponse::InternalServerError().body(format!("Generation failed: {}", e)),
    }
}

#[actix_web::main]
#[cfg(feature = "server")]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    let host = "127.0.0.1:8080";
    println!("Starting server at http://{}", host);
    HttpServer::new(|| App::new().service(generate)).bind(host)?.run().await
}

#[cfg(not(feature = "server"))]
fn main() {
    println!("server example disabled; enable with `--features server` to run the example");
}
