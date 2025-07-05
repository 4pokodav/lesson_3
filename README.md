# Домашнее задание к уроку 3: Полносвязные сети
## Задание 1: Эксперименты с глубиной сети (30 баллов)

### 1.1 Сравнение моделей разной глубины (15 баллов)
Создал и обучил модели с различным количеством слоев:
- 1 слой (линейный классификатор) | (train acc - 0.9842; test acc - 0.9756; training time - 74.38 sec)
- 2 слоя (1 скрытый) | (train acc - 0.9841; test acc - 0.9749; training time - 77.07)
- 3 слоя (2 скрытых) | (train acc - 0.9828; test acc - 0.9716; training time - 79.47)
- 5 слоев (4 скрытых) | (train acc - 0.9801; test acc - 0.9737; training time - 79.22)
- 7 слоев (6 скрытых) | (train acc - 0.979; test acc - 0.9769; training time - 80.79)

Модель с 1 слоем:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_1.png)

Модель с 2 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_2.png)

Модель с 3 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_3.png)

Модель с 5 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_5.png)

Модель с 7 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_7.png)

### 1.2 Анализ переобучения (15 баллов)
Исследовал влияние глубины на переобучение:
- Построил графики train/test accuracy по эпохам
- Добавил Dropout и BatchNorm, результаты оказались похожие.
- В основном, переобучения не было. Только у модели с глубиной 7 после 8 эпохи начилось переобучение.

Модель с 1 слоем:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_1.png)

Модель с 2 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_2.png)

Модель с 3 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_3.png)

Модель с 5 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_5.png)

Модель с 7 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_7.png)

## Задание 2: Эксперименты с шириной сети (25 баллов)

### 2.1 Сравнение моделей разной ширины (15 баллов)
Создал модели с различной шириной слоев, но с фиксированной глубиной (3 слоя):
- Узкие слои: [64, 32, 16] | (train acc - 0.9759; test acc - 0.9614; training time - 77.58 sec)
- Средние слои: [256, 128, 64] | (train acc - 0.9867; test acc - 0.9786; training time - 76.91 sec)
- Широкие слои: [1024, 512, 256] | (train acc - 0.9866; test acc - 0.9783; training time - 99.98 sec)
- Очень широкие слои: [2048, 1024, 512] | (train acc - 0.; test acc - 0.; training time -  sec)

Количество параметров возрастает примерно в 5 раз при каждом увеличении ширины. У последней модели примерно 4.5 млн параметров.

Модель с узкими слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_narrow.png)

Модель с средними слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_medium.png)

Модель с широкими слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_wide.png)

Модель с очень широкими слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_very_wide.png)

### 2.2 Оптимизация архитектуры (10 баллов)

Найдите оптимальную архитектуру:
- Использовал grid search для поиска лучшей комбинации
- Визуализиализировал результаты в виде heatmap

![Heatmap](https://github.com/4pokodav/lesson_3/raw/main/plots/grid_search_heatmap.png)

## Задание 3: Эксперименты с регуляризацией (25 баллов)

### 3.1 Сравнение техник регуляризации (15 баллов)

Исследовал различные техники регуляризации:
- Без регуляризации
- Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
- Только BatchNorm
- Dropout + BatchNorm
- L2 регуляризация (weight decay)

Графики распределения весов приложены.

### 3.2 Адаптивная регуляризация (10 баллов)

Реализовал адаптивные техники:
- Dropout с изменяющимся коэффициентом
- BatchNorm с различными momentum
- Комбинирование нескольких техник
