# Учебный проект по ускорению инференса super-resolution модели

Этот репозиторий содержит midterm-этап курсового проекта по ускорению инференса U-Net-подобной модели для задачи single-image super-resolution.

## О чем проект

Цель проекта — получить **честный baseline** для super-resolution, а затем сравнить несколько inference-пайплайнов по:
- latency
- throughput
- качеству восстановления изображения

В качестве quality-метрик используются:
- PSNR
- SSIM

## Зафиксированная постановка

На текущем этапе выбраны следующие инженерные решения:
- задача: single-image super-resolution
- модель: усиленная U-Net-подобная encoder-decoder архитектура со skip-connections
- scale factor: x2
- деградация: bicubic downsampling из HR в LR
- вход модели: bicubic-upsampled LR-изображение
- target: исходное HR-изображение

Такой setup выбран специально, чтобы:
- быстро получить воспроизводимый baseline;
- не тратить слишком много времени на сложную подготовку данных;
- сосредоточиться на основной цели проекта — ускорении инференса.

## Базовая модель

В проекте используется U-Net-подобная модель для super-resolution:
- входной projection layer
- encoder-decoder со skip-connections
- три уровня downsampling / upsampling
- residual prediction поверх входного bicubic baseline

Реализация модели находится в [`src/modeling.py`](efficient-ml-midterm/src/modeling.py).

## Данные

Проект ожидает HR-изображения в следующей структуре:

```text
./data/sr/
  train_hr/
  val_hr/
  test_hr/
```

LR-изображения не хранятся отдельно, а генерируются **на лету** из HR через bicubic downsampling внутри датасет-пайплайна.

Реализация датасета находится в [`src/data.py`](efficient-ml-midterm/src/data.py).

## Что уже реализовано

В репозитории уже есть:
- датасет для SR
- модель baseline
- обучение baseline
- оценка качества
- benchmark для FP32
- benchmark для FP16
- шаблон таблицы результатов
- черновик отчета для midterm

## Структура репозитория

- [`src/`](efficient-ml-midterm/src) — код датасета, модели, обучения, оценки и бенчмарка
- [`configs/`](efficient-ml-midterm/configs) — конфиги экспериментов
- [`scripts/`](efficient-ml-midterm/scripts) — shell-скрипты запуска
- [`results/`](efficient-ml-midterm/results) — результаты и сравнительные таблицы
- [`docs/`](efficient-ml-midterm/docs) — отчет и материалы для презентации

## Планируемые inference-пайплайны

После получения baseline планируется сравнить:
1. PyTorch baseline FP16
2. `torch.compile`
3. альтернативный компилятор, например TVM
4. квантизацию
5. pruning / 2:4 semi-structured sparsity
6. profiling и ablation study

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Основные конфиги

- [`configs/sr_baseline_x2.yaml`](efficient-ml-midterm/configs/sr_baseline_x2.yaml) — основной baseline
- [`configs/sr_baseline_x2_fp16.yaml`](efficient-ml-midterm/configs/sr_baseline_x2_fp16.yaml) — FP16 benchmark

## Скрипты запуска

Подготовка структуры данных:

```bash
bash scripts/prepare_sr_data.sh
```

Обучение baseline:

```bash
bash scripts/train_sr_baseline.sh
```

Оценка качества:

```bash
bash scripts/eval_sr.sh
```

Бенчмарк FP32:

```bash
bash scripts/benchmark_sr_fp32.sh
```

Бенчмарк FP16:

```bash
bash scripts/benchmark_sr_fp16.sh
```

## Что смотреть во время обучения

Во время обучения в консоль печатается:
- номер эпохи
- средний `train_l1`

После обучения нужно смотреть:
- PSNR
- SSIM

После этого отдельно измеряются:
- latency
- throughput

## Что является первым результатом проекта

Первый обязательный результат для midterm:
1. обученный baseline
2. PSNR / SSIM на validation
3. FP32 benchmark
4. FP16 benchmark
5. заполненный отчет и таблица результатов

## Замечание

Текущий репозиторий ориентирован именно на **первый воспроизводимый baseline**. После этого на его основе будут строиться все дальнейшие эксперименты по ускорению инференса.
