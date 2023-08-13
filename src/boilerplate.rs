use std::sync::Arc;

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanLibrary, image::{ImageUsage, SwapchainImage},
};
use winit::window::Window;

pub fn create_instance() -> Arc<vulkano::instance::Instance> {
    // A loaded library containing a valid Vulkan implementation.
    let library = VulkanLibrary::new().unwrap();

    // What extensions do we need in order to draw to a window?
    let required_extensions = vulkano_win::required_extensions(&library);
    println!("{:?}", required_extensions);
    println!("required_extensions: OK");

    // Create a Vulkan
    return Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            // Enable enumerating devices that use non-conformant Vulkan implementations. (e.g.
            // MoltenVK)
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();
}

// The function is named get_device_and_family_index and takes three parameters:
//      instance: An Arc (atomic reference counted smart pointer) to a Vulkan instance. The Vulkan instance represents the entry 
//          point to the Vulkan API and is required for various operations.
//      device_extensions: A set of device extensions that the physical device should support.
//      surface: An Arc to a Vulkan surface, which represents the window or surface onto which graphics will be rendered.

// The function returns a tuple containing:
// An Arc to the selected physical device (suitable for the application).
// An unsigned integer representing the index of the queue family that supports graphics operations and can present images to the specified surface.

// The function performs the following steps:

// It calls the enumerate_physical_devices method on the Vulkan instance to retrieve a list of physical devices available on the system.

// It filters the list of physical devices using the filter method. Each physical device is checked to ensure that it supports 
// the specified device_extensions.

// It then uses the filter_map method to further filter the physical devices based on the queue families they offer. For each physical device:
//     It iterates over the queue family properties using queue_family_properties() method.
//     For each queue family, it checks if it supports graphics operations (as indicated by the GRAPHICS flag) and if it can present images 
//          to the given surface. If both conditions are met, the index of the queue family is retained along with the physical device.

// After filtering, it uses the min_by_key method to select the physical device with the lowest score. The score is assigned based 
// on the device type, with certain types being preferred over others. The min_by_key function will select the physical device with the lowest score.

// The score assignment is done using a match expression on the physical device's type. Lower scores are assigned to device types that are 
// likely to be faster or better suited for graphics operations. The device types are:
// DiscreteGpu: Dedicated graphics card
// IntegratedGpu: Integrated graphics
// VirtualGpu: Virtualized graphics
// Cpu: CPU-based graphics
// Other: Other types of devices
// Any other type: Assigned a default score of 5

// Finally, the expect method is used to handle the case where no suitable physical device is found. An error message is displayed
// if no suitable device is available.

pub fn get_device_and_family_index(
    instance: Arc<vulkano::instance::Instance>,
    device_extensions: DeviceExtensions,
    surface: &Arc<Surface>,
) -> (Arc<PhysicalDevice>, u32) {
    return instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // Find queues we need for graphics
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");
}

// It takes three parameters:

// physical_device: An Arc (smart pointer) to a Vulkan PhysicalDevice object, representing a physical GPU.
// device_extensions: An object specifying the device extensions that need to be enabled for the Vulkan device.
// queue_family_index: An index indicating the desired queue family for the device.

// Device::new is called with the chosen PhysicalDevice and a DeviceCreateInfo struct that specifies various device-related parameters.
// The enabled_extensions field of DeviceCreateInfo is set to the provided device_extensions.
// The queue_create_infos field is set to a vector containing a single QueueCreateInfo object. This indicates that a 
// queue from the specified queue family should be created.
// The other fields of DeviceCreateInfo are set to their default values using Rust's ..Default::default() syntax.
// The Device::new method returns a Result object, which is then unwrapped using the .unwrap() method. This will either 
// return the created Vulkan device or panic if there was an error during device creation.

// The function returns a tuple containing:

// An Arc to the created Vulkan Device.
// An iterator representing the queue(s) associated with the created device. The type of the iterator is impl ExactSizeIterator<Item = Arc<Queue>>.


pub fn get_device_and_queues(
    physical_device: Arc<PhysicalDevice>,
    device_extensions: DeviceExtensions,
    queue_family_index: u32,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    return Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            // A list of optional features and extensions that our program needs to work correctly.
            // Some parts of the Vulkan specs are optional and must be enabled manually at device
            // creation. In this example the only thing we are going to need is the `khr_swapchain`
            // extension that allows us to draw to a window.
            enabled_extensions: device_extensions,

            // The list of queues that we are going to use. Here we only use one queue, from the
            // previously chosen queue family.
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],

            ..Default::default()
        },
    )
    .unwrap();
}

pub fn get_swapchain_and_image(device: &Arc<Device>, surface: &Arc<Surface>) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {   
    // Before we can draw on the surface, we have to create what is called a swapchain. Creating a
    // swapchain allocates the color buffers that will contain the image that will ultimately be
    // visible on the screen. These images are returned alongside the swapchain.
    // let (mut swapchain, images) = {
        return {
        // Querying the capabilities of the surface. When we create the swapchain we can only pass
        // values that are allowed by the capabilities.
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        // Choosing the internal format that the images will have.
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );
        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,

                image_format,

                // The dimensions of the window, only used to initially setup the swapchain.
                //
                // NOTE:
                // On some drivers the swapchain dimensions are specified by
                // `surface_capabilities.current_extent` and the swapchain size must use these
                // dimensions. These dimensions are always the same as the window dimensions.
                //
                // However, other drivers don't specify a value, i.e.
                // `surface_capabilities.current_extent` is `None`. These drivers will allow
                // anything, but the only sensible value is the window dimensions.
                //
                // Both of these cases need the swapchain to use the window dimensions, so we just
                // use that.
                image_extent: window.inner_size().into(),

                image_usage: ImageUsage::COLOR_ATTACHMENT,

                // The alpha mode indicates how the alpha value of the final image will behave. For
                // example, you can choose whether the window will be opaque or transparent.
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )
        .unwrap()
    };


}

