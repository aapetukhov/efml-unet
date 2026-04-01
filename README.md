# EFML Super-Resolution (Hydra Template)

Переработанная структура с Hydra-конфигами, датасетами, тренером и скриптами `train.py` / `inference.py`, адаптированная под задачу single-image super-resolution (x2) на данных из `data/sr`.

## Структура
```
.
├── requirements.txt
├── src
│   ├── configs
│   │   ├── baseline.yaml          # общий тренинг-конфиг (Hydра)
│   │   ├── inference.yaml         # конфиг инференса/оценки
│   │   ├── dataloader/sr.yaml     # батч, num_workers, pin_memory
│   │   ├── datasets/sr.yaml       # пути train/val/test HR, scale, crop
│   │   ├── metrics/sr.yaml        # список метрик
│   │   ├── model/unet.yaml        # параметры U-Net
│   │   ├── transforms/sr.yaml     # нормализация и resize опции
│   │   └── writer/console.yaml    # логгер/трэкер
│   ├── datasets                   # датасеты и коллейты
│   ├── logger                     # логгирование + заглушка wandb
│   ├── loss                       # функции потерь
│   ├── metrics                    # SSIM + трекер
│   ├── model                      # U-Net baseline
│   ├── trainer                    # base_trainer / trainer / inferencer
│   ├── transforms                 # normalize/scale helpers
│   └── utils                      # seed/device, I/O вспомогательные утилиты
├── train.py                       # запуск обучения через Hydra
└── inference.py                   # запуск инференса/оценки
```

## Данные
Ожидаемая структура:
```
data/sr/
  train_hr/
  val_hr/
  test_hr/
```
LR генерируются на лету из HR через bicubic downsample → bicubic upscale.

## Запуск
Установить зависимости:
```bash
pip install -r requirements.txt
```

Обучение (использует `src/configs/baseline.yaml`):
```bash
python train.py trainer.device=auto datasets.scale=2
```

Инференс/оценка (берёт чекпоинт из `checkpoints/sr_unet_x2.pt` по умолчанию):
```bash
python inference.py checkpoint_path=checkpoints/sr_unet_x2.pt save_predictions=true
```
Hydra позволяет переопределять любые параметры, например `dataloader.batch_size=4` или `datasets.train.crop_size=128` без копипаста конфигов.

## Ключевые отличия от старой версии
- Шаблонная директория и конфигурации в стиле ASR-DeepSpeech2, без дублирования настроек между экспериментами.
- Один базовый Trainer + Inferencer с поддержкой mixed precision, grad clipping и простого логгинга.
- Метрика SSIM вынесена в отдельный пакет и задаётся через конфиги.
- Датасет генерирует LR/HR пары на лету, есть отдельные конфиги для датасетов и даталоадеров.
- Writer по умолчанию console; можно переключить на wandb через `writer.name=wandb` (при установленном `wandb`).
