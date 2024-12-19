from model import SIRENa

if __name__ == "__main__":
    model = SIRENa()
    img_path = "/path/to/img"
    save_dir = "/path/to/save/dir"
    model(img_path, save_dir)
