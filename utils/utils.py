def display_local_video_in_notebook(filename, size=(600, 400)):
    try:
        get_ipython()
        from IPython.display import HTML
    except (NameError, ImportError):
        raise RuntimeError("`display_local_video_in_notebook` must be called from jupyter/colab.")

    import base64
    import os
    # Re-encode video to x264.
    os.system(f"ffmpeg -y -i {filename} -vcodec libx264 {filename}.x264.mp4 -hide_banner -loglevel error")
    os.replace(filename + ".x264.mp4", filename)
    # Convert to base64 for display in notebook.
    with open(filename, "rb") as f:
        video_data = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
    display(
        HTML(f"""
            <video width="{size[0]}" height="{size[1]}" controls autoplay loop>
                <source type="video/mp4" src="{video_data}">
                Your browser does not support the video tag.
            </video>
        """))