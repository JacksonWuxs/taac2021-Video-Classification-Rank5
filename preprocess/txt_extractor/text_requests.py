# -*- coding: utf-8 -*-

class VideoASR():
    """视频ASR"""
    def request(self, inp):
        return inp
    
class VideoOCR():
    """视频OCR"""
    def request(self, inp):
        return inp
    
class ImageOCR():
    """图像OCR"""
    def request(self, inp):
        return inp
   
if __name__ == '__main__':
    test_image = './test.jpg'
    image_ocr = ImageOCR().request(test_image)
    print("image_ocr: {}".format(image_ocr))

