from app import app
from flask import request, render_template
from keras import models
import numpy as np
from PIL import Image
import string
import random
import os


# adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# loading a model
model = models.load_model('app/static/model/model55.h5')


# route to home page
@app.route('/', methods=['GET', 'POST'])
def index():
    # if request is GET
    if request.method == 'GET':
        full_filename = 'images/Pictogrammers-Material-Light-Camera.512.png'
        return render_template('index2.html', full_filename=full_filename)

    # if request is POST
    if request.method == 'POST':
        # generate unique image name
        letters = string.ascii_lowercase
        name = ''.join(random.choice(letters) for _ in range(10)) + '.png'
        full_filename = 'uploads/' + name
        # reading, resizing, saving and preprocessing for prediction
        image_upload = request.files['fileup']
        # image_name = image_upload.filename
        image = Image.open(image_upload)
        image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))
        image = image.resize((150, 150))

        image_arr = np.array(image.convert('RGB'))
        image_arr.shape = (1, 150, 150, 3)

        # predicting outputs
        result = model.predict(image_arr)
        ind = np.argmax(result)
        classes = ['A-1', 'A-11', 'A-11a', 'A-12a', 'A-14', 'A-15', 'A-16', 'A-17', 'A-18b', 'A-2', 'A-20', 'A-21',
                   'A-24', 'A-29', 'A-3', 'A-30', 'A-32', 'A-4', 'A-6a', 'A-6b', 'A-6c', 'A-6d', 'A-6e', 'A-7', 'A-8',
                   'B-1', 'B-18', 'B-2', 'B-20', 'B-21', 'B-22', 'B-25', 'B-26', 'B-27', 'B-33', 'B-34', 'B-36', 'B-41',
                   'B-42', 'B-43', 'B-44', 'B-5', 'B-6-B-8-B-9', 'B-8', 'B-9', 'C-10', 'C-12', 'C-13', 'C-13-C-16',
                   'C-13a', 'C-13a-C-16a', 'C-16', 'C-2', 'C-4', 'C-5', 'C-6', 'C-7', 'C-9', 'D-1', 'D-14', 'D-15',
                   'D-18', 'D-18b', 'D-2', 'D-21', 'D-23', 'D-23a', 'D-24', 'D-26', 'D-26b', 'D-26c', 'D-27', 'D-28',
                   'D-29', 'D-3', 'D-40', 'D-41', 'D-42', 'D-43', 'D-4a', 'D-4b', 'D-51', 'D-52', 'D-53', 'D-6', 'D-6b',
                   'D-7', 'D-8', 'D-9', 'D-tablica', 'G-1a', 'G-3']

        class_names = {
            'A-1': 'Niebezpieczny zakręt w prawo',
            'A-11': 'Nierówna droga',
            'A-11a': 'Próg zwalniający',
            'A-12a': 'Zwężenie jezdni - dwustronne',
            'A-14': 'Roboty drogowe',
            'A-15': 'Śliska jezdnia',
            'A-16': 'Przejście dla pieszych',
            'A-17': 'Dzieci',
            'A-18b': 'Zwierzęta dzikie',
            'A-2': 'Niebezpieczny zakręt w lewo',
            'A-20': 'Odcinek jezdni o ruchu dwukierunkowym',
            'A-21': 'Tramwaj',
            'A-24': 'Rowerzyści',
            'A-29': 'Sygnały świetlne',
            'A-3': 'Niebezpieczne zakręty, pierwszy w prawo',
            'A-30': 'Inne niebezpieczeństwo',
            'A-32': 'Oszronienie jezdni',
            'A-4': 'Niebezpieczne zakręty, pierwszy w lewo',
            'A-6a': 'Skrzyżowanie z drogą podporządkowaną występującą po obu stronach',
            'A-6b': 'Skrzyżowanie z drogą podporządkowaną występującą po prawej stronie',
            'A-6c': 'Skrzyżowanie z drogą podporządkowaną występującą po lewej stronie',
            'A-6d': 'Wlot drogi jednokierunkowej z prawej strony',
            'A-6e': 'Wlot drogi jednokierunkowej z lewej strony',
            'A-7': 'Ustąp pierwszeństwa',
            'A-8': 'Skrzyżowanie o ruchu okrężnym',
            'B-1': 'Zakaz ruchu w obu kierunkach',
            'B-18': 'Zakaz wjazdu pojazdów o rzeczywistej masie całkowitej ponad ... t.',
            'B-2': 'Zakaz wjazdu',
            'B-20': 'STOP',
            'B-21': 'Zakaz skręcania w lewo',
            'B-22': 'Zakaz skręcania w prawo',
            'B-25': 'Zakaz wyprzedzania',
            'B-26': 'Zakaz wyprzedzania przez samochody ciężarowe',
            'B-27': 'Koniec zakazu wyprzedzania',
            'B-33': 'Ograniczenie prędkości',
            'B-34': 'Koniec ograniczenia prędkości',
            'B-36': 'Zakaz zatrzymywania się',
            'B-41': 'Zakaz ruchu pieszych',
            'B-42': 'Koniec zakazów',
            'B-43': 'Strefa ograniczonej prędkości',
            'B-44': 'Koniec strefy ograniczonej prędkości',
            'B-5': 'Zakaz wjazdu samochodów ciężarowych',
            'B-6-B-8-B-9': 'Zakaz wjazdu pojazdów innych niż samochodowe',
            'B-8': 'Zakaz wjazdu pojazdów zaprzęgowych',
            'B-9': 'Zakaz wjazdu rowerów',
            'C-10': 'Nakaz jazdy z lewej strony znaku',
            'C-12': 'Ruch okrężny',
            'C-13': 'Droga dla rowerów',
            'C-13-C-16': 'Droga dla pieszych i rowerzystów',
            'C-13a': 'Koniec drogi dla rowerów',
            'C-13a-C-16a': 'Koniec drogi dla pieszych i rowerzystów',
            'C-16': 'Droga dla pieszych',
            'C-2': 'Nakaz jazdy w prawo za znakiem',
            'C-4': 'Nakaz jazdy w lewo za znakiem',
            'C-5': 'Nakaz jazdy prosto',
            'C-6': 'Nakaz jazdy prosto lub w prawo',
            'C-7': 'Nakaz jazdy prosto lub w lewo',
            'C-9': 'Nakaz jazdy z prawej strony znaku',
            'D-1': 'Droga z pierwszeństwem',
            'D-14': 'Koniec pasa ruchu',
            'D-15': 'Przystanek autobusowy',
            'D-18': 'Parking',
            'D-18b': 'Parking zadaszony',
            'D-2': 'Koniec drogi z pierwszeństwem',
            'D-21': 'Szpital',
            'D-23': 'Stacja paliwowa',
            'D-23a': 'Stacja paliwowa tylko z gazem do napędu pojazdów',
            'D-24': 'Telefon',
            'D-26': 'Stacja obsługi technicznej',
            'D-26b': 'Myjnia',
            'D-26c': 'Toaleta publiczna',
            'D-27': 'Bufet lub kawiarnia',
            'D-28': 'Restauracja',
            'D-29': 'Hotel (motel)',
            'D-3': 'Droga jednokierunkowa',
            'D-40': 'Strefa zamieszkania',
            'D-41': 'Koniec strefy zamieszkania',
            'D-42': 'Obszar zabudowany',
            'D-43': 'Koniec obszaru zabudowanego',
            'D-4a': 'Droga bez przejazdu',
            'D-4b': 'Wjazd na drogę bez przejazdu',
            'D-51': 'Automatyczna kontrola prędkości',
            'D-52': 'Strefa ruchu',
            'D-53': 'Koniec strefy ruchu',
            'D-6': 'Przejście dla pieszych',
            'D-6b': 'Przejście dla pieszych i przejazd dla rowerzystów',
            'D-7': 'Droga ekspresowa',
            'D-8': 'Koniec drogi ekspresowej',
            'D-9': 'Autostrada',
            'D-tablica': 'Zbiorcza tablica informacyjna',
            'G-1a': 'Słupek wskaźnikowy z trzema kreskami umieszczany po prawej stronie jezdni',
            'G-3': 'Krzyż św. Andrzeja przed przejazdem kolejowym jednotorowym'
        }

        return render_template('index2.html', full_filename=full_filename, pred_num=classes[ind],
                               pred_name=class_names[classes[ind]])

    if __name__ == '__main__':
        app.run(debug=True)
