import os
import matplotlib.pyplot as plt
from .config import TRAIN_DIR, VAL_DIR, TEST_DIR

def analyze_dataset():
    """
    Проверяет количество изображений в классах и строит график распределения.
    """
    categories = ['WithMask', 'WithoutMask']
    dirs = {'Train': TRAIN_DIR, 'Validation': VAL_DIR, 'Test': TEST_DIR}
    
    print("=== Анализ Датасета ===")
    
    stats = {}
    
    for split_name, split_path in dirs.items():
        stats[split_name] = []
        for category in categories:
            path = os.path.join(split_path, category)
            if not os.path.exists(path):
                print(f"⚠️ Путь не найден: {path}")
                count = 0
            else:
                count = len(os.listdir(path))
            stats[split_name].append(count)
            print(f"[{split_name}] {category}: {count} изображений")

    # Визуализация баланса классов (для отчета)
    # Если нужно показать преподавателю - раскомментируй plt.show()
    labels = categories
    x = range(len(labels))
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, (split_name, counts) in enumerate(stats.items()):
        ax[i].bar(labels, counts, color=['green', 'red'])
        ax[i].set_title(f'Распределение {split_name}')
        ax[i].set_ylabel('Количество')
    
    plt.savefig('dataset_analysis.png')
    print("График распределения сохранен как dataset_analysis.png")
    print("=======================")

if __name__ == "__main__":
    analyze_dataset()