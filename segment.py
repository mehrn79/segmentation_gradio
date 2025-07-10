from monai.bundle import ConfigParser

def segmentor():

    # مسیر به فایل config
    config_path = "/media/external20/mehran_advand/gradio/monai_wholeBody_ct_segmentation/configs/inference.json"

    # بارگذاری config
    parser = ConfigParser()
    parser.read_config(config_path)

    # گرفتن evaluator از config
    evaluator = parser.get_parsed_content("evaluator")

    # اجرای امن با کنترل خطا
    try:
        evaluator.run()
    except Exception as e:
        print(f"⚠️ خطا در اجرای evaluator: {e}")


