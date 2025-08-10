import os
import vtk

def read_and_render_obj(obj_file_path, output_image_path):
    # Read the OBJ file
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(obj_file_path)

    # Set up the mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(obj_reader.GetOutputPort())

    # Set up the actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 1, 1) 

    # Set up the renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)


    # Set up the render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 800)  # Window size
    render_window.AddRenderer(renderer)

    # Render the scene
    render_window.OffScreenRenderingOn()
    render_window.Render()

    # Set up the window to image filter
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetScale(2)  # Increase resolution of the output image
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.ReadFrontBufferOff()
    window_to_image_filter.Update()

    # Write the image to a PNG file
    image_writer = vtk.vtkPNGWriter()
    image_writer.SetFileName(output_image_path)
    image_writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    image_writer.Write()

def process_directory(input_dir, output_dir):
    # Traverse the directory structure
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".obj"):
                # Full path to the .obj file
                obj_file_path = os.path.join(root, file)

                # Create the corresponding output directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Create the output image file path by replacing .obj with .png
                image_file_name = file.replace(".obj", ".png")
                output_image_path = os.path.join(output_subdir, image_file_name)

                # Render and save the image
                read_and_render_obj(obj_file_path, output_image_path)
                print(f"Processed {obj_file_path} -> {output_image_path}")


if __name__ == "__main__":

    # Input and output directories
    input_directory = "/home/shirshak/Teeth3DS_individual_teeth/individual_teeth/" 
    output_directory = "/home/shirshak/Teeth3DS_individual_teeth/individual_teeth_thumbnail/"  

    # Process the directory
    process_directory(input_directory, output_directory)