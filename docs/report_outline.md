# Черновик midterm-отчета

## 1. Состав команды
- Имя 1
- Имя 2
- Имя 3

## 2. Цель проекта
Ускорить инференс U-Net-подобной модели для задачи single-image super-resolution и сравнить latency, throughput и качество для нескольких inference-пайплайнов на одной и той же модели и одном и том же датасете.

## 3. Зафиксированная постановка
- Задача: single-image super-resolution
- Модель: легкая U-Net-подобная encoder-decoder архитектура со skip-connections
- Датасет: открытый набор HR-изображений, разложенный на train / val / test
- Scale factor: x2
- Генерация LR: bicubic downsampling из HR, затем bicubic upsampling обратно до размера HR для подачи в модель
- Метрика качества: SSIM

## 4. Первый эксперимент
- Подготовить train и validation split
- Генерировать LR-входы на лету из HR-изображений
- Обучить baseline-модель
- Посчитать SSIM на validation
- Замерить PyTorch inference в FP32 и FP16

## 5. План всех экспериментов
1. PyTorch baseline FP16
2. `torch.compile`
3. TVM или другой альтернативный компилятор
4. Квантизация
5. Pruning / structured sparsity
6. Profiling и ablation study

## 6. Понедельный план работы
- Неделя 1: подготовка датасета, реализация baseline-модели, первый train/eval pipeline
- Неделя 2: честный baseline, FP32/FP16 benchmark, черновик отчета
- Неделя 3: `torch.compile`, profiling, первые ablation-эксперименты
- Неделя 4: quantization, pruning, optional TVM, итоговая таблица и презентация

## 7. Результаты первого эксперимента
Заполняется после первого запуска.

Что сюда внести:
- размеры train / val / test split
- scale factor и способ генерации LR
- число эпох, optimizer, learning rate
- validation SSIM
- validation SSIM
- FP32 latency / throughput
- FP16 latency / throughput

## 8. Воспроизводимость
- GitHub-репозиторий
- [`requirements.txt`](efficient-ml-midterm/requirements.txt)
- скрипты обучения, оценки и бенчмарка
- фиксированный конфиг первого эксперимента
