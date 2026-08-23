//! Cross-platform GPU presentation for the native Dream64 client.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use wgpu::util::{DeviceExt, TextureBlitter};
use winit::window::Window;

/// A `wgpu` surface that presents Dream64's authoritative client frame.
pub(crate) struct GpuRenderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    configuration: wgpu::SurfaceConfiguration,
    frame_texture: wgpu::Texture,
    blitter: TextureBlitter,
    sprite_pipeline: wgpu::RenderPipeline,
    sprite_bind_group_layout: wgpu::BindGroupLayout,
    sprite_sampler: wgpu::Sampler,
    dmi_atlas: DmiAtlas,
    adapter_label: String,
}

/// One ordered, already-rasterized appearance submitted to the GPU compositor.
pub(crate) struct SpriteDraw {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u32>,
    pub(crate) clip: [u32; 4],
}

/// One raw DMI cell rendered from a persistent full-sheet GPU atlas.
pub(crate) struct DmiSpriteDraw {
    pub(crate) resource: PathBuf,
    pub(crate) sheet_width: u32,
    pub(crate) sheet_height: u32,
    pub(crate) rgba: Arc<[u8]>,
    pub(crate) source: [u32; 4],
    pub(crate) destination: [f32; 4],
    pub(crate) tint: [u8; 4],
    pub(crate) clip: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpriteInstance {
    destination: [f32; 4],
    atlas_uv: [f32; 4],
    clip: [f32; 4],
    viewport: [f32; 2],
    tint: [f32; 4],
}

struct DmiAtlas {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    next_x: u32,
    next_y: u32,
    row_height: u32,
    entries: HashMap<PathBuf, AtlasRegion>,
}

#[derive(Clone, Copy)]
struct AtlasRegion {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl GpuRenderer {
    /// Creates the preferred GPU renderer for the current platform.
    pub(crate) fn new(window: Arc<Window>) -> Result<Self, String> {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: preferred_backends(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| format!("create GPU surface: {error}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|error| format!("request GPU adapter: {error}"))?;
        let info = adapter.get_info();
        let adapter_label = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Dream64 client GPU"),
                ..Default::default()
            })
            .await
            .map_err(|error| format!("request GPU device: {error}"))?;
        let mut configuration = surface
            .get_default_config(&adapter, width, height)
            .ok_or("GPU adapter cannot present to the client window")?;
        configuration.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &configuration);
        let frame_texture = create_frame_texture(&device, width, height);
        let blitter = TextureBlitter::new(&device, configuration.format);
        let (sprite_pipeline, sprite_bind_group_layout, sprite_sampler) =
            create_sprite_pipeline(&device, configuration.format);
        let dmi_atlas = DmiAtlas::new(
            &device,
            &sprite_bind_group_layout,
            &sprite_sampler,
            adapter.limits().max_texture_dimension_2d.min(8_192),
        );
        Ok(Self {
            window,
            surface,
            device,
            queue,
            configuration,
            frame_texture,
            blitter,
            sprite_pipeline,
            sprite_bind_group_layout,
            sprite_sampler,
            dmi_atlas,
            adapter_label,
        })
    }

    pub(crate) fn window(&self) -> &Window {
        &self.window
    }

    pub(crate) fn adapter_label(&self) -> &str {
        &self.adapter_label
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.configuration.width = width;
        self.configuration.height = height;
        self.surface.configure(&self.device, &self.configuration);
        self.frame_texture = create_frame_texture(&self.device, width, height);
    }

    /// Uploads and presents one top-down BGRA8 client framebuffer.
    pub(crate) fn present(
        &mut self,
        pixels: &[u32],
        dmi_sprites: &[DmiSpriteDraw],
        sprites: &[SpriteDraw],
    ) -> Result<(), String> {
        let expected = usize::try_from(self.configuration.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.configuration.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or("GPU framebuffer dimensions overflow")?;
        if pixels.len() != expected {
            return Err(format!(
                "GPU framebuffer has {} pixels; expected {expected}",
                pixels.len()
            ));
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.frame_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.configuration.width.saturating_mul(4)),
                rows_per_image: Some(self.configuration.height),
            },
            wgpu::Extent3d {
                width: self.configuration.width,
                height: self.configuration.height,
                depth_or_array_layers: 1,
            },
        );
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.configuration);
                self.surface
                    .get_current_texture()
                    .map_err(|error| format!("recover GPU surface: {error}"))?
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(error) => return Err(format!("acquire GPU surface: {error}")),
        };
        let source = self
            .frame_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dream64 frame encoder"),
            });
        self.blitter
            .copy(&self.device, &mut encoder, &source, &target);
        if !dmi_sprites.is_empty() {
            self.draw_dmi_sprites(&mut encoder, &target, dmi_sprites)?;
        }
        if !sprites.is_empty() {
            self.draw_sprites(&mut encoder, &target, sprites)?;
        }
        self.queue.submit([encoder.finish()]);
        frame.present();
        Ok(())
    }

    fn draw_sprites(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        sprites: &[SpriteDraw],
    ) -> Result<(), String> {
        let atlas = build_atlas(sprites, self.configuration.width, self.configuration.height)?;
        static BATCH_DIAGNOSTIC: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        BATCH_DIAGNOSTIC.get_or_init(|| {
            eprintln!(
                "client-gpu-sprite-batch: sprites={} atlas={}x{}",
                sprites.len(),
                atlas.width,
                atlas.height
            );
        });
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dream64 per-frame sprite atlas"),
            size: wgpu::Extent3d {
                width: atlas.width,
                height: atlas.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&atlas.pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas.width * 4),
                rows_per_image: Some(atlas.height),
            },
            wgpu::Extent3d {
                width: atlas.width,
                height: atlas.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dream64 sprite atlas bind group"),
            layout: &self.sprite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sprite_sampler),
                },
            ],
        });
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dream64 ordered sprite instances"),
                contents: bytemuck::cast_slice(&atlas.instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Dream64 ordered sprite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.sprite_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, instance_buffer.slice(..));
        pass.draw(
            0..6,
            0..u32::try_from(atlas.instances.len()).unwrap_or(u32::MAX),
        );
        Ok(())
    }

    fn draw_dmi_sprites(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        sprites: &[DmiSpriteDraw],
    ) -> Result<(), String> {
        let mut instances = Vec::with_capacity(sprites.len());
        for sprite in sprites {
            let region = self.dmi_atlas.ensure_sheet(&self.queue, sprite)?;
            instances.push(SpriteInstance {
                destination: sprite.destination,
                atlas_uv: [
                    region.x.saturating_add(sprite.source[0]) as f32 / self.dmi_atlas.width as f32,
                    region.y.saturating_add(sprite.source[1]) as f32 / self.dmi_atlas.height as f32,
                    sprite.source[2] as f32 / self.dmi_atlas.width as f32,
                    sprite.source[3] as f32 / self.dmi_atlas.height as f32,
                ],
                clip: sprite.clip.map(|value| value as f32),
                viewport: [
                    self.configuration.width as f32,
                    self.configuration.height as f32,
                ],
                tint: sprite.tint.map(|value| f32::from(value) / 255.0),
            });
        }
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dream64 persistent DMI instances"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Dream64 persistent DMI sprite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.sprite_pipeline);
        pass.set_bind_group(0, &self.dmi_atlas.bind_group, &[]);
        pass.set_vertex_buffer(0, instance_buffer.slice(..));
        pass.draw(0..6, 0..u32::try_from(instances.len()).unwrap_or(u32::MAX));
        Ok(())
    }
}

impl DmiAtlas {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        dimension: u32,
    ) -> Self {
        let dimension = dimension.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dream64 persistent DMI atlas"),
            size: wgpu::Extent3d {
                width: dimension,
                height: dimension,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dream64 persistent DMI atlas bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            texture,
            bind_group,
            width: dimension,
            height: dimension,
            next_x: 0,
            next_y: 0,
            row_height: 0,
            entries: HashMap::new(),
        }
    }

    fn ensure_sheet(
        &mut self,
        queue: &wgpu::Queue,
        sprite: &DmiSpriteDraw,
    ) -> Result<AtlasRegion, String> {
        if let Some(region) = self.entries.get(&sprite.resource) {
            if (region.width, region.height) != (sprite.sheet_width, sprite.sheet_height) {
                return Err(format!(
                    "DMI dimensions changed after GPU upload: {}",
                    sprite.resource.display()
                ));
            }
            return Ok(*region);
        }
        let expected = usize::try_from(sprite.sheet_width)
            .ok()
            .and_then(|width| {
                usize::try_from(sprite.sheet_height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or("DMI sheet dimensions overflow")?;
        if sprite.rgba.len() != expected {
            return Err(format!(
                "DMI sheet {} has {} bytes; expected {expected}",
                sprite.resource.display(),
                sprite.rgba.len()
            ));
        }
        if sprite.sheet_width > self.width || sprite.sheet_height > self.height {
            return Err(format!(
                "DMI sheet {} ({}x{}) exceeds {}x{} atlas",
                sprite.resource.display(),
                sprite.sheet_width,
                sprite.sheet_height,
                self.width,
                self.height
            ));
        }
        if self.next_x > 0 && self.next_x.saturating_add(sprite.sheet_width) > self.width {
            self.next_y = self.next_y.saturating_add(self.row_height);
            self.next_x = 0;
            self.row_height = 0;
        }
        if self.next_y.saturating_add(sprite.sheet_height) > self.height {
            return Err("persistent DMI atlas is full".to_owned());
        }
        let region = AtlasRegion {
            x: self.next_x,
            y: self.next_y,
            width: sprite.sheet_width,
            height: sprite.sheet_height,
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x,
                    y: region.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &sprite.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(sprite.sheet_width * 4),
                rows_per_image: Some(sprite.sheet_height),
            },
            wgpu::Extent3d {
                width: sprite.sheet_width,
                height: sprite.sheet_height,
                depth_or_array_layers: 1,
            },
        );
        eprintln!(
            "client-gpu-dmi-upload: resource={} sheet={}x{} atlas={},{}",
            sprite.resource.display(),
            sprite.sheet_width,
            sprite.sheet_height,
            region.x,
            region.y
        );
        self.next_x = self.next_x.saturating_add(sprite.sheet_width);
        self.row_height = self.row_height.max(sprite.sheet_height);
        self.entries.insert(sprite.resource.clone(), region);
        Ok(region)
    }
}

struct SpriteAtlas {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
    instances: Vec<SpriteInstance>,
}

fn build_atlas(
    sprites: &[SpriteDraw],
    viewport_width: u32,
    viewport_height: u32,
) -> Result<SpriteAtlas, String> {
    const MAX_ATLAS_WIDTH: u32 = 4_096;
    let widest = sprites.iter().map(|sprite| sprite.width).max().unwrap_or(1);
    let combined_width = sprites
        .iter()
        .fold(0_u32, |sum, sprite| sum.saturating_add(sprite.width));
    let width = widest.max(combined_width.min(MAX_ATLAS_WIDTH)).max(1);
    if width > MAX_ATLAS_WIDTH {
        return Err(format!("sprite width {width} exceeds GPU atlas limit"));
    }
    let mut placements = Vec::with_capacity(sprites.len());
    let (mut x, mut y, mut row_height) = (0_u32, 0_u32, 0_u32);
    for sprite in sprites {
        if sprite.width > MAX_ATLAS_WIDTH {
            return Err(format!(
                "sprite width {} exceeds GPU atlas limit",
                sprite.width
            ));
        }
        if x > 0 && x.saturating_add(sprite.width) > width {
            y = y.saturating_add(row_height);
            x = 0;
            row_height = 0;
        }
        placements.push((x, y));
        x = x.saturating_add(sprite.width);
        row_height = row_height.max(sprite.height);
    }
    let height = y.saturating_add(row_height).max(1);
    let pixel_count = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or("GPU atlas dimensions overflow")?;
    let mut pixels = vec![0_u32; pixel_count];
    let mut instances = Vec::with_capacity(sprites.len());
    for (sprite, (atlas_x, atlas_y)) in sprites.iter().zip(placements) {
        let source_width = usize::try_from(sprite.width).map_err(|_| "sprite width overflow")?;
        let expected_pixels = source_width
            .checked_mul(usize::try_from(sprite.height).map_err(|_| "sprite height overflow")?)
            .ok_or("sprite dimensions overflow")?;
        if sprite.pixels.len() != expected_pixels {
            return Err(format!(
                "sprite has {} pixels; expected {expected_pixels}",
                sprite.pixels.len()
            ));
        }
        for row in 0..sprite.height {
            let source_start = usize::try_from(row)
                .ok()
                .and_then(|row| row.checked_mul(source_width))
                .ok_or("sprite row overflow")?;
            let destination_start = usize::try_from(atlas_y + row)
                .ok()
                .and_then(|row| row.checked_mul(usize::try_from(width).ok()?))
                .and_then(|offset| offset.checked_add(usize::try_from(atlas_x).ok()?))
                .ok_or("atlas row overflow")?;
            pixels[destination_start..destination_start + source_width]
                .copy_from_slice(&sprite.pixels[source_start..source_start + source_width]);
        }
        instances.push(SpriteInstance {
            destination: [
                sprite.x as f32,
                sprite.y as f32,
                sprite.width as f32,
                sprite.height as f32,
            ],
            atlas_uv: [
                atlas_x as f32 / width as f32,
                atlas_y as f32 / height as f32,
                sprite.width as f32 / width as f32,
                sprite.height as f32 / height as f32,
            ],
            clip: sprite.clip.map(|value| value as f32),
            viewport: [viewport_width as f32, viewport_height as f32],
            tint: [1.0; 4],
        });
    }
    Ok(SpriteAtlas {
        width,
        height,
        pixels,
        instances,
    })
}

fn create_sprite_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Dream64 sprite bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Dream64 pixel sprite sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Dream64 sprite pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dream64 ordered sprite shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("sprite.wgsl").into()),
    });
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
        3 => Float32x2,
        4 => Float32x4
    ];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Dream64 ordered sprite pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SpriteInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &ATTRIBUTES,
            }],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });
    (pipeline, bind_group_layout, sampler)
}

fn create_frame_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dream64 authoritative client frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

const fn preferred_backends() -> wgpu::Backends {
    #[cfg(target_os = "windows")]
    {
        return wgpu::Backends::DX12.union(wgpu::Backends::VULKAN);
    }
    #[cfg(target_os = "linux")]
    {
        return wgpu::Backends::VULKAN.union(wgpu::Backends::GL);
    }
    #[cfg(target_os = "macos")]
    {
        return wgpu::Backends::METAL;
    }
    #[allow(unreachable_code)]
    wgpu::Backends::PRIMARY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_backend_policy_is_not_empty() {
        assert!(!preferred_backends().is_empty());
    }

    #[test]
    fn atlas_preserves_order_pixels_and_destinations() {
        let first = [0xff00_0001, 0xff00_0002];
        let second = [0xff00_0003, 0xff00_0004];
        let sprites = [
            SpriteDraw {
                x: 4,
                y: 5,
                width: 2,
                height: 1,
                pixels: first.to_vec(),
                clip: [0, 0, 64, 64],
            },
            SpriteDraw {
                x: 8,
                y: 9,
                width: 1,
                height: 2,
                pixels: second.to_vec(),
                clip: [0, 0, 64, 64],
            },
        ];
        let atlas = build_atlas(&sprites, 64, 64).unwrap();
        assert_eq!(atlas.instances.len(), 2);
        assert_eq!(atlas.instances[0].destination, [4.0, 5.0, 2.0, 1.0]);
        assert_eq!(atlas.instances[1].destination, [8.0, 9.0, 1.0, 2.0]);
        assert!(atlas.pixels.contains(&0xff00_0001));
        assert!(atlas.pixels.contains(&0xff00_0004));
    }
}
