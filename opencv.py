#LAB: Clasificación de imágenes con Watson VR y Python.

import cv2
import urllib.request
from matplotlib import pyplot as plt
from pylab import rcParams
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
from pandas.io.json import json_normalize

def plt_image(image_url, size=(10, 8)):
    # Descarga una imagen desde una URL y la muestra en el cuaderno

    urllib.request.urlretrieve(image_url, "image.jpg")  # downloads file as "image.jpg"
    image = cv2.imread("image.jpg")

    # Si la imagen es a color, corrige de BGR a RGB la codificación

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rcParams['figure.figsize'] = size[0], size[1]  # set image display size

    plt.axis("off")
    plt.imshow(image, cmap="Greys_r")
    plt.show()

#image_url = 'http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Images/Donald_Trump_Justin_Trudeau_2017-02-13_02.jpg'
#plt_image(image_url)

# Pega aquí abajo la API key de IBM Watson Visual Recognition:

my_apikey = '7_fPX5vxcsijEg5wRBre3TL5SjX9cLMu7h-gwZP_t0C2'

authenticator = IAMAuthenticator(my_apikey)

visrec = VisualRecognitionV3('2018-03-19',
                             authenticator=authenticator)

#image_url = 'http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Images/Donald_Trump_Justin_Trudeau_2017-02-13_02.jpg'

# El valor del argumento threshold es igual a 0.6, solo las clases que tienen un valor de confianza de a 0.6 o mayor será mostrado en pantalla

#classes = visrec.classify(url=image_url,
#                          threshold=0.6).get_result()

#plt_image(image_url)
#print(json.dumps(classes, indent=2))

def getdf_visrec(url, apikey=my_apikey):
    json_result = visrec.classify(url=url, threshold=0.6).get_result()

    json_classes = json_result['images'][0]['classifiers'][0]['classes']

    df = json_normalize(json_classes).sort_values('score', ascending=False).reset_index(drop=True)

    return df


#url = 'http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Images/76011_MAIN._AC_SS190_V1446845310_.jpg'
#plt_image(url)
#print(getdf_visrec(url))

url = 'http://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Images/2880px-Kyrenia_01-2017_img04_view_from_castle_bastion.jpg'
plt_image(url)
print(getdf_visrec(url))