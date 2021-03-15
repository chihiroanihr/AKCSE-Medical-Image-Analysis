from PIL import Image
import glob

def formatIm(currDir):
    c=1
    for filename in glob.glob(currDir + 'dataset-raw/*.*'): #assuming gif
        im=Image.open(filename)
        name='img'+str(c)+'.png'
        format_im = im.convert('RGBA')
        format_im = im.resize((512,512), Image.ANTIALIAS)
        format_im.save(currDir + 'dataset-formatted/'+name)
        c+=1 
    print ("images resized and reformatted")

formatIm('c:/Users/hamsi/OneDrive/Desktop/AKCSE-Medical-Image-Analysis/')