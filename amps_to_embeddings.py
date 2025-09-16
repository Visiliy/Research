import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import time
import gc
import signal
import sys

# Глобальный флаг для graceful shutdown
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    print("\n[INFO] Interrupt received, saving progress and exiting...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

# Включаем expandable_segments (если поддерживается)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ========================

def read_file_batch(file_paths):
    """Чтение батча файлов с фильтрацией пустых"""
    results = []
    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                content = fh.read().strip()
                if content:
                    results.append((str(fp), content))
        except Exception as e:
            continue
    return results

def tokenize_batch(tokenizer, batch_texts):
    """Токенизация батча текстов"""
    return [tokenizer.encode(text).ids for text in batch_texts]

def read_texts_from_folder(folder):
    """Быстрое получение списка файлов"""
    p = Path(folder)
    if not p.exists():
        return []
    return list(p.rglob('*.txt'))

def process_files_parallel(files, batch_size=5000, max_workers=None):
    """Оптимизированное параллельное чтение"""
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)
    
    file_paths = [str(f) for f in files]
    all_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            futures.append(executor.submit(read_file_batch, batch))
        
        for i, future in enumerate(as_completed(futures)):
            try:
                batch_results = future.result()
                if batch_results:
                    all_results.extend(batch_results)
            except Exception as e:
                continue
            if i % 10 == 0:
                print(f"Processed {len(all_results)} files")
            if interrupted:
                break
    
    return all_results

def train_tokenizer_sequential(texts, vocab_size=50000):
    """Обучение токенизатора на уже прочитанных текстах"""
    temp_file = "temp_tokenizer_texts.txt"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[temp_file], 
            vocab_size=vocab_size, 
            min_frequency=2, 
            special_tokens=['<s>', '</s>', '<pad>', '<unk>', '<mask>']
        )
        return tokenizer
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def tokenize_sequential(tokenizer, texts, batch_size=1000):
    """Последовательная токенизация"""
    all_indices = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        if interrupted:
            break
        batch_texts = texts[i:i + batch_size]
        batch_indices = [tokenizer.encode(text).ids for text in batch_texts]
        all_indices.extend(batch_indices)
    return all_indices

class EfficientTextEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    
    def forward(self, x):
        return self.emb(x)

def get_positional_encoding(seq_len, dim, device):
    """Векторизованное вычисление позиционных эмбеддингов (вызывается один раз)"""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * 
                        (-np.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# ========================
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ
# ========================

def process_gpu_batches(model, indices, file_paths, batch_size, device, emb_dim, save_dir="embeddings"):
    """Обработка на GPU с немедленным сохранением каждого батча на диск — НИЧЕГО НЕ ДЕРЖИМ В ПАМЯТИ"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Загружаем список уже обработанных файлов
    processed_indices = set()
    embedding_files = sorted([f for f in os.listdir(save_dir) if f.startswith("batch_") and f.endswith(".pt")])
    for fname in embedding_files:
        try:
            start_idx = int(fname.split("_")[1])
            end_idx = int(fname.split("_")[2].split(".")[0])
            for i in range(start_idx, end_idx):
                processed_indices.add(i)
        except:
            continue

    if len(processed_indices) > 0:
        print(f"Resuming from {len(processed_indices)} processed files")

    # Предварительно вычисляем позиционные эмбеддинги
    max_len = 512
    pos_emb = get_positional_encoding(max_len, emb_dim, device)
    
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    total_processed = len(processed_indices)
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(indices), batch_size), desc="GPU Processing", initial=total_processed//batch_size):
            if interrupted:
                break

            start_idx = i
            end_idx = min(i + batch_size, len(indices))
            
            # Пропускаем уже обработанные
            if all(idx in processed_indices for idx in range(start_idx, end_idx)):
                continue

            batch_indices = indices[start_idx:end_idx]
            batch_files = file_paths[start_idx:end_idx]
            
            # Подготовка батча с паддингом
            batch_tensors = []
            for seq in batch_indices:
                if len(seq) > max_len:
                    seq = seq[:max_len]
                else:
                    seq = seq + [0] * (max_len - len(seq))
                batch_tensors.append(seq)
            
            x = torch.tensor(batch_tensors, dtype=torch.long, device=device)
            
            # Прямой проход
            text_emb = model(x)
            combined_emb = text_emb + pos_emb.unsqueeze(0)  # Broadcasting
            
            # Сохраняем на диск сразу — НЕ ДЕРЖИМ В ПАМЯТИ!
            batch_output_path = os.path.join(save_dir, f"batch_{start_idx}_{end_idx}.pt")
            torch.save({
                'embeddings': combined_emb.cpu(),  # Переносим на CPU перед сохранением
                'files': batch_files,
                'start_idx': start_idx,
                'end_idx': end_idx
            }, batch_output_path)
            
            # Очистка
            del x, text_emb, combined_emb
            torch.cuda.empty_cache()
            gc.collect()
            
            # Лог памяти GPU
            if device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU Memory: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")

            total_processed = end_idx
            elapsed = time.time() - start_time
            speed = total_processed / elapsed if elapsed > 0 else 0
            print(f"  Saved batch {start_idx}-{end_idx} ({total_processed}/{len(indices)} files, {speed:.1f} files/s)")

    # После завершения — создаем индексный файл
    all_files = []
    all_embeddings = []  # Только если нужно объединить в конце — иначе не используйте!

    # Собираем список файлов и опционально объединяем эмбеддинги (осторожно с памятью!)
    embedding_files = sorted([f for f in os.listdir(save_dir) if f.startswith("batch_") and f.endswith(".pt")],
                             key=lambda x: int(x.split("_")[1]))
    
    for fname in embedding_files:
        data = torch.load(os.path.join(save_dir, fname), map_location='cpu')
        all_files.extend(data['files'])
        # Если нужно объединить — раскомментируйте, но рискуете OOM:
        # all_embeddings.append(data['embeddings'])
    
    # Если вы хотите объединить в один файл — делайте это ТОЛЬКО если памяти достаточно
    # if all_embeddings:
    #     final_embeddings = torch.cat(all_embeddings, dim=0)
    #     torch.save({
    #         'embeddings': final_embeddings,
    #         'files': all_files
    #     }, os.path.join(save_dir, "all_embeddings.pt"))
    #     print(f"Final embeddings saved: {final_embeddings.shape}")

    print(f"✅ All batches processed and saved to '{save_dir}'")
    return None, all_files  # Эмбеддинги не возвращаем — они на диске

# ========================
# MAIN
# ========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    folder = 'amps'
    files = read_texts_from_folder(folder)
    
    if not files:
        print("No files found")
        return
    
    print(f"Found {len(files):,} files")
    
    # Этап 1: Чтение файлов
    start_time = time.time()
    file_data = process_files_parallel(files, batch_size=5000)
    
    if not file_data:
        print("No valid text content found")
        return
    
    file_paths = [data[0] for data in file_data]
    texts = [data[1] for data in file_data]
    
    read_time = time.time() - start_time
    print(f"Reading completed: {len(texts):,} files in {read_time:.1f}s ({len(texts)/read_time:.1f} files/s)")
    
    # Этап 2: Обучение токенизатора
    print("Training tokenizer...")
    tokenizer = train_tokenizer_sequential(texts)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer trained with vocab size: {vocab_size}")
    tokenizer.save("tokenizer.json")
    
    # Этап 3: Токенизация
    print("Tokenizing texts...")
    tokenize_time = time.time()
    indices = tokenize_sequential(tokenizer, texts, batch_size=1000)
    tokenize_time = time.time() - tokenize_time
    print(f"Tokenization completed in {tokenize_time:.1f}s")
    
    if interrupted:
        print("🛑 Stopped by user during tokenization")
        return

    # Этап 4: Эмбеддинги на GPU
    emb_dim = 512
    model = EfficientTextEmbedding(vocab_size, emb_dim).to(device)
    
    # ⚠️ Ключевое: уменьшаем батч до 32 или 64 для 8GB GPU
    gpu_batch_size = 32  # Было 256 — слишком много!
    
    print(f"Starting GPU processing with batch_size={gpu_batch_size}")
    _, processed_files = process_gpu_batches(
        model, indices, file_paths, gpu_batch_size, device, emb_dim, save_dir="amps_embeddings"
    )
    
    total_time = time.time() - start_time
    print(f"✅ Processing completed: {len(processed_files):,} files in {total_time:.1f}s")
    print(f"📊 Average speed: {len(processed_files)/total_time:.1f} files/s")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise