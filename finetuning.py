import finetuner
from finetuner import CSVOptions

finetuner.login()

run = finetuner.fit(
    model='openai/clip-vit-base-patch32',
    train_data='./data/dilbert/character_annotations.csv',
    csv_options=CSVOptions(is_labeled=True),
    run_name='clip-run',
    loss='CLIPLoss',
    epochs=5,
    learning_rate=1e-6,
)
