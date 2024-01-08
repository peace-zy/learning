GRADIO_TEMP_DIR = "tmp"
if not os.path.exists(GRADIO_TEMP_DIR):
    os.makedirs(GRADIO_TEMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
