## Немного о моих изменениях написал в самом низу!

# Задача

**Требуется:** предложить модель, сегментирующую человека на фотографии.

**Вход:** фотография 320x240x3.
**Выход:** маска человека 320x240.
**Метрика:** [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).

# Данные

Ссылка на скачивание данных: [link](https://yadi.sk/d/lSkJ25yjP0t8tQ).

Данные представляют из себя набор фотографий человека и маски, определяющей положение человека на фотографии.

Доступные данные разделены на несколько папок:
- `train` содержит фотографии 320x240x3;
- `train_mask` содержит маски для фотографий из `train` 320x240;
- `valid` содержит фотографии 320x240x3;
- `valid_mask` содержит маски для фотографий из `valid` 320x240;
- `test` содержит фотографии 320x240x3.

# Результаты

Для лучшей модели требуется создать 2 файла, которые необходимы для валидации Вашего решения:
- сохраненные маски для картинок из valid в формате pred_valid_template.csv (в архиве с `data`);
- html страницу с предсказанием модели для всех картинок из test и папку с используемыми картинками в этой html странице для её просмотра.

Также необходимо:
- подготовить код для проверки (докстринги, PEP8);
- создать отчет (в jupyter ноутбуке) с описанием Вашего исследования, предобработки, постобработки, проверямых гипотез, используемых моделей, описание лучшего подхода и т.п.

# Рекомендуемый pipeline решения:

Предполагается следующий pipeline решения поставленной задачи:
- скачать данные;
- ознакомиться с `notebooks/GettingStarted.ipynb`;
- провести анализ данных;
- реализовать нейросеть/нейросети (предполагается реализация "с нуля");
- Обучить реализованные модели с использованием best practices;
- провалидировать модели;
- выбрать **лучшую** модель на `valid` и посчитать для нее метрики и создать необходимые файлы (см. `notebooks/GettingStarted.ipynb`);
- отправить результаты на проверку.

# Критерии оценивания

1. Качество и полнота исследования в jupyter notebook.
2. Анализ данных, подходы к аугментации и пред- и постобработки. Оригинальность, эффективность и количество идей. Использование неочевидных шагов в решении, которые улучшают качество, будет хорошим плюсом.
3. Чистота кода.
3. Структура кода.
3. Значение метрик на контрольной выборке и общая адекватность модели.
4. Обоснование выбора моделей (процесс выбора моделей).

# Что я сделал

Привет!
Спасибо за интересную задачку, было весело заниматься. 

Я решил реализовать PSPNet, потому что модель очень хорошо работает, быстро обучается и прилично пишется по сравнению с другими моделями сегментации. Ну по крайней мере так показалось :)

Раз нам нужно было максимизировать Dice, то в качестве лосса выбрал аппроксимацию к нему. Работало неплохо, по скорости так же как и BCELoss, но зато оптимизировало сразу правильную задачу. 

Для аугментации изображений использовал Albumentations — удобная библиотека, в которой можно аугментировать изображения для задачи сегментации!

Модель обучалась на Kaggle: еле влез в 15 гигабайт, но смог. 

В своей работе я не трогал изначальный код в `lib/`, но добавлял туда свои функции в существующие файлы или создавал новые. Ко всем старался добавить документацию.
