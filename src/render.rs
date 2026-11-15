// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

use std::sync::Arc;

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;

use crate::tiles::{GpuTiles, MAX_LEVELS};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub vp: [f32; 4],
    pub img: [f32; 4],
    pub cam: [f32; 4],
    pub tile: [f32; 4],
    pub flags: [u32; 4],
    pub grid: [f32; 4],
    pub grid_color: [f32; 4],
    pub bg: [f32; 4],
    /// Per pyramid level: (nx | ny << 16, first page-table row, width, height).
    pub levels: [[u32; 4]; MAX_LEVELS],
}

impl Uniforms {
    pub fn set_levels(&mut self, tiles: &GpuTiles) {
        for (dst, l) in self.levels.iter_mut().zip(&tiles.grid.levels) {
            *dst = [l.nx | (l.ny << 16), l.row, l.w, l.h];
        }
    }
}

pub struct PanoRenderer {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    samp_lin: wgpu::Sampler,
    bind_group: Option<(u32, wgpu::BindGroup)>,
}

impl PanoRenderer {
    pub fn install(rs: &egui_wgpu::RenderState) -> bool {
        let device = &rs.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pano shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pano.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pano bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pano layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pano pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: rs.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pano uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mk = |f: wgpu::FilterMode, m: wgpu::MipmapFilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("pano sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: f,
                min_filter: f,
                mipmap_filter: m,
                ..Default::default()
            })
        };
        let samp_lin = mk(wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear);
        let encode_srgb = !rs.target_format.is_srgb();
        rs.renderer.write().callback_resources.insert(PanoRenderer {
            pipeline,
            bgl,
            uniforms,
            samp_lin,
            bind_group: None,
        });
        encode_srgb
    }
}

pub struct PanoCallback {
    pub uniforms: Uniforms,
    pub pano: Option<Arc<GpuTiles>>,
}

impl CallbackTrait for PanoCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _sd: &ScreenDescriptor,
        _enc: &mut wgpu::CommandEncoder,
        res: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = res.get_mut::<PanoRenderer>() else { return Vec::new() };
        if let Some(p) = &self.pano {
            if r.bind_group.as_ref().map(|b| b.0) != Some(p.id) {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pano bind group"),
                    layout: &r.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: r.uniforms.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&p.view) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&r.samp_lin) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&p.table_view) },
                    ],
                });
                r.bind_group = Some((p.id, bg));
            }
        } else {
            r.bind_group = None;
        }
        queue.write_buffer(&r.uniforms, 0, bytemuck::bytes_of(&self.uniforms));
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        res: &CallbackResources,
    ) {
        let Some(r) = res.get::<PanoRenderer>() else { return };
        let Some((_, bg)) = &r.bind_group else { return };
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
