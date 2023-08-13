// #![allow(unused_imports)]
// #![allow(unused_variables)]
// #![allow(unused_mut)]

mod boilerplate;
use std::sync::Arc;

use crate::boilerplate::{create_instance, get_device_and_family_index, get_device_and_queues, get_swapchain_and_image};

use vulkano::{device::DeviceExtensions, memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryUsage}, pipeline::{graphics::{vertex_input::Vertex, input_assembly::InputAssemblyState, viewport::{ViewportState, Viewport}}, GraphicsPipeline}, buffer::{BufferContents, Buffer, BufferCreateInfo, BufferUsage}, render_pass::{Subpass, RenderPass, Framebuffer, FramebufferCreateInfo}, image::{SwapchainImage, ImageAccess, view::ImageView}, swapchain::{SwapchainPresentInfo, SwapchainCreateInfo, SwapchainCreationError, acquire_next_image, AcquireError}, command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, RenderPassBeginInfo, SubpassContents, CommandBufferUsage}, sync::{self, GpuFuture, FlushError}};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event_loop::{EventLoop, ControlFlow},
    window::{WindowBuilder, Window}, event::{Event, WindowEvent},
};

fn main() {
    // ----------------------- INSTANCE
    let instance = create_instance();
    println!("instance created: OK");

    // ----------------------- EVENT LOOP
    let event_loop = EventLoop::new();
    println!("event_loop created: OK");

    // ----------------------- SWAP-CHAIN
    let device_extensions = DeviceExtensions { khr_swapchain: true,  ..DeviceExtensions::empty() };
    println!("{:?}", device_extensions);
    println!("device_extensions: OK");

    // ----------------------- SURFACE
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    println!("surface created: OK");

    // ----------------------- PHYSICAL DEVICE and QUEUE FAMILY INDEX
    let (physical_device, queue_family_index) = get_device_and_family_index(instance, device_extensions, &surface);
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // ----------------------- LOGICAL DEVICE and QUEUES
    let (device, mut queues) = 
        get_device_and_queues(physical_device.clone(), device_extensions, queue_family_index);
    println!("device created: OK");
    println!("queues created: OK");
    let queue = queues.next().unwrap();

    // ----------------------- SWAPCHAIN and IMAGES
    let (mut swapchain, images) = get_swapchain_and_image(&device, &surface);
    println!("swapchain: OK");
    println!("images: OK");
 

    // ---------------------------------------------------------------------------
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // We now create a buffer that will store the shape of our triangle. We use `#[repr(C)]` here
    // to force rustc to use a defined layout for our data, as the default representation has *no
    // guarantees*.

    // The BufferContents trait is responsible for specifying how the data of a Rust struct should be laid out 
    // in a GPU buffer. This involves specifying the alignment, size, and layout of the data fields.

    // The Vertex trait, on the other hand, defines how the vertex attributes in your Rust struct correspond 
    // to the input variables in your vertex shader. It involves specifying the names of the shader inputs, 
    // their formats, and any special requirements.

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
        },
        Vertex {
            position: [0.0, 0.5],
        },
        Vertex {
            position: [0.25, -0.1],
        },
    ];
    
    let vertex_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    // --------------- SHADERS

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ",
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // --------------- RENDERPASS

    // The next step is to create a *render pass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images where the
    // colors, depth and/or stencil information will be written.
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // `load: Clear` means that we ask the GPU to clear the content of this attachment
                // at the start of the drawing.
                load: Clear,
                // `store: Store` means that we ask the GPU to store the output of the draw in the
                // actual image. We could also ask it to discard the result.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image. This has to be one
                // of the types of the `vulkano::format` module (or alternatively one of your
                // structs that implements the `FormatDesc` trait). Here we use the same format as
                // the swapchain.
                format: swapchain.image_format(),
                // `samples: 1` means that we ask the GPU to use one sample to determine the value
                // of each pixel in the color attachment. We could use a larger value
                // (multisampling) for antialiasing. An example of this can be found in
                // msaa-renderpass.rs.
                samples: 1,
            },
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            // No depth-stencil attachment is indicated with empty brackets.
            depth_stencil: {},
        },
    )
    .unwrap();

    // --------------------- PIPELINE and VIEWPORT

    // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
    // program, but much more specific.
    let pipeline = GraphicsPipeline::start()
        // We have to indicate which subpass of which render pass this pipeline is going to be used
        // in. The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        // We need to indicate the layout of the vertices.
        .vertex_input_state(Vertex::per_vertex())
        // The content of the vertex buffer describes a list of triangles.
        .input_assembly_state(InputAssemblyState::new())
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one.
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        // Use a resizable viewport set to draw over the entire window
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        // See `vertex_shader`.
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .unwrap();

    // Dynamic viewports allow us to recreate just the viewport when the window is resized.
    // Otherwise we would have to recreate the whole pipeline.
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    



    // -------------------- FRAMEBUFFERS and COMMAND BUFFERS

        // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    // Before we can start creating and recording command buffers, we need a way of allocating
    // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command pools
    // underneath and provides a safe interface for them.
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain. Here,
    // we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // Do not draw the frame when the screen dimensions are zero. On Windows, this can
                // occur when minimizing the application.
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                // It is important to call this function from time to time, otherwise resources
                // will keep accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU
                // has already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if recreate_swapchain {
                    // Use the new dimensions of the window.

                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the
                            // window. Simply restarting the loop is the easiest way to fix this
                            // issue.
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("failed to recreate swapchain: {e}"),
                        };

                    swapchain = new_swapchain;

                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );

                    recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the
                // swapchain. If no image is available (which happens if you submit draw commands
                // too quickly), then the function will block. This operation returns the index of
                // the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional
                // timeout after which the function call will return an error.
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                // In order to draw, we have to build a *command buffer*. The command buffer object
                // holds the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*.
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // A list of values to clear the attachments with. This list contains
                            // one item for each attachment in the render pass. In this case, there
                            // is only one attachment, and we clear it with a blue color.
                            //
                            // Only attachments that have `LoadOp::Clear` are provided with clear
                            // values, any others should use `ClearValue::None` as the clear value.
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        // The contents of the first (and only) subpass. This can be either
                        // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more advanced
                        // and is not covered here.
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    //
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    // We add a draw command.
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // We leave the render pass. Note that if we had multiple subpasses we could
                    // have called `next_subpass` to jump to the next subpass.
                    .end_render_pass()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });



}


/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}