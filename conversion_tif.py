from PIL import Image, ImageSequence
import os
import os.path
import glob

#należy uzupełnić ścieżkę do katalogu z plikami
files = glob.glob('...')

for file in files:
    filename_ext = os.path.basename(file)
    filename = os.path.splitext(filename_ext)[0]
    try:
        im = Image.open(file)
        for i, page in enumerate(ImageSequence.Iterator(im)):
            #należy uzupełnić ścieżkę do katalogu, gdzie mają być zapisane pliki
            path = "..." + filename + "-"+str(i+1)+".png"
            if not os.path.isfile(path):
                try:
                    page.save(path)
                except:
                    print(filename_ext)
    except:
        print(filename_ext)
