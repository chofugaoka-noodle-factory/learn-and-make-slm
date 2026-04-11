from argparse import ArgumentParser
import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import tiktoken
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from config import SLMConfig
from dataset import SimpleStoriesBatchTorch, SimpleStoriesBothDataset, dataset_collate
from models.transformer.decoder import GPT2Decoder, GPTOSSDecoder
from utils.checkpoint import latest_checkpoint_path, load_checkpoint, save_checkpoint
from utils.model import get_model
from utils.tools import to_device

if TYPE_CHECKING:
    from models.transformer.decoder import DecoderBase


def validate(
    model: "DecoderBase",
    test_loader: torch.utils.data.DataLoader,
    test_writer: SummaryWriter,
    tokenizer: tiktoken.Encoding,
    epoch: int | None = None,
    step: int | None = None,
):
    assert (epoch is not None and step is None) or (epoch is None and step is not None), (
        f"Exactly one of 'epoch' or 'step' must be provided (not both, not neither). Got: epoch={epoch!r}, step={step!r}"
    )

    model.eval()

    all_loss = 0
    count = 0

    with torch.no_grad():
        for _batch in tqdm(test_loader, desc="Validation", dynamic_ncols=True, position=2):
            batch: SimpleStoriesBatchTorch = to_device(_batch, next(model.parameters()).device)
            seq_lens = batch["lengths"] - 1

            with torch.amp.autocast("cuda", enabled=False):
                output, _, _ = model.forward(batch["tokens_ids"][:, :-1], seq_lens=seq_lens)
                loss = model.loss(output, batch["tokens_ids"][:, 1:], seq_lens)
            all_loss += loss.item()

            if epoch is not None and count < 5:
                for i in range(min(batch["tokens_ids"].size(0), 5 - count)):
                    tokens_ids = batch["tokens_ids"][i : i + 1]
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        output_tokens_ids = model.infer(starts=tokens_ids[:, :5], max_token_count=200)
                    gt_text = tokenizer.decode(tokens_ids[0, : batch["lengths"][i]].tolist())
                    output_text = tokenizer.decode(output_tokens_ids[0].tolist())
                    test_writer.add_text(f"Text/{i}/GT", gt_text, 0)
                    test_writer.add_text(f"Text/{i}/Output", output_text, epoch)
                count += min(batch["tokens_ids"].size(0), 5 - count)

    test_writer.add_scalar(
        f"Loss/{'Epoch' if epoch is not None else 'Step'}_Test",
        all_loss / len(test_loader),
        epoch if epoch is not None else step,
    )
    model.train()


def train(
    config: SLMConfig,
    model: "DecoderBase",
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    tokenizer: tiktoken.Encoding,
    start_epoch: int,
    enable_amp: bool = True,
):
    # TensorBoardでlossや学習率の推移を記録するためのWriterを初期化
    train_log_dir = config.path.log_dir
    test_log_dir = os.path.join(config.path.log_dir, "test")
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)

    train_writer = SummaryWriter(log_dir=train_log_dir)
    test_writer = SummaryWriter(log_dir=test_log_dir)

    # 学習時に用いた設定を保存
    with open(os.path.join(train_log_dir, "config.yaml"), "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False, allow_unicode=True)

    # float16を使って計算を高速化する仕組み(Mixed Precision Training: AMP/混合精度訓練)を設定
    # GradScalerは勾配のスケーリングを自動で行い、数値の安定性を保つ
    use_amp = enable_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if not use_amp:
        tqdm.write("[INFO] AMP (mixed precision) disabled")

    # 全体の進捗を表示するためのバー
    train_steps = len(train_loader) // config.train.gradient_accumulation_steps
    if len(train_loader) % config.train.gradient_accumulation_steps != 0:
        train_steps += 1
    total_bar = tqdm(
        total=config.train.epochs * train_steps,
        position=0,
        desc="Steps",
        dynamic_ncols=True,
    )
    # チェックポイントから再開する場合は、既に終わったステップ数を設定
    total_bar.n = (start_epoch - 1) * train_steps

    # 学習率スケジューラ、エポックごとに学習率を減衰させる(lr_decay倍)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_decay, last_epoch=start_epoch - 2)

    interrupted = False
    # Gradient Accumulation用の変数
    accumulation_step = 0  # 現在何個目のバッチを累積しているか
    accumulated_loss = 0.0  # 累積中のloss合計(ログ用)

    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_loss = 0
        for batch_step, _batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, position=1)):
            try:
                # データをGPUに転送
                batch: SimpleStoriesBatchTorch = to_device(_batch, next(model.parameters()).device)

                # === Gradient Accumulationサイクルの開始 ===
                # 累積サイクルの最初だけ勾配をゼロクリア
                # （途中のバッチでは勾配を累積するためクリアしない）
                if accumulation_step == 0:
                    optimizer.zero_grad()

                # === 1. 順伝播 (Forward Pass) ===
                # モデルに入力を与えて、lossを計算する
                seq_lens = batch["lengths"] - 1
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output, _, _ = model(batch["tokens_ids"][:, :-1], seq_lens=seq_lens)
                    loss = model.loss(output, batch["tokens_ids"][:, 1:], seq_lens)

                # === 2. 逆伝播 (Backward Pass) ===
                # loss.backward()で勾配を計算し、param.gradに加算
                # この時点では重みは更新されず、勾配が計算・蓄積されるだけ
                scaler.scale(loss).backward()

                # ログ用にlossを記録
                accumulated_loss += loss.item()
                epoch_loss += loss.item()

                accumulation_step += 1

                # === 3. 重みの更新 (Weight Update) ===
                # 指定回数分の勾配を累積するか、エポックの最後のバッチなら重みを更新
                # 勾配蓄積を利用することで、バッチサイズが小さくても実質的なバッチサイズを大きくできて、学習を安定させることができる
                if accumulation_step == config.train.gradient_accumulation_steps or batch_step == len(train_loader) - 1:
                    # scalerから実際の勾配値を取り出す(mixed precision用)
                    scaler.unscale_(optimizer)

                    # 累積した回数で割り、勾配を正規化する
                    # backward()は勾配を加算するので、平均を取らないと学習率が大きくなりすぎる
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad /= accumulation_step

                    # 今回はなくても良かったが、勾配爆発を防ぐために勾配クリッピングを行うこともある
                    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # optimizer.step()を呼んで初めてパラメータが更新される
                    scaler.step(optimizer)
                    scaler.update()

                    # ログ出力
                    avg_loss = accumulated_loss / accumulation_step

                    if total_bar.n % config.train.logging_steps == 0:
                        # tqdm.write(f"Step {total_bar.n}, Loss: {avg_loss:.4f}, Grad Norm: {grad_norm:.4f}")
                        tqdm.write(f"Step {total_bar.n}, Loss: {avg_loss:.4f}")
                        train_writer.add_scalar("Loss/Step_Train", avg_loss, total_bar.n)
                        # train_writer.add_scalar("Grad_Norm", grad_norm.item(), total_bar.n)
                        # if total_bar.n != 0:
                        #     validate(model, test_loader, test_writer, tokenizer, step=total_bar.n)

                    total_bar.update()

                    # 次の累積サイクルのためにリセット
                    accumulation_step = 0
                    accumulated_loss = 0.0

            except KeyboardInterrupt:
                if interrupted:
                    raise KeyboardInterrupt
                tqdm.write("KeyboardInterrupt detected. Please wait for the checkpoint to be saved...")
                tqdm.write(
                    "If you want to quit immediately, please press Ctrl+C again. But the checkpoint will not be saved."
                )
                interrupted = True

        # === エポック終了時の処理 ===

        # エポック全体の平均lossをTensorBoardに記録
        train_writer.add_scalar("Loss/Epoch_Train", epoch_loss / len(train_loader), epoch)

        # 現在の学習率をTensorBoardに記録
        train_writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        # 学習率スケジューラーを更新(学習率を減衰)
        scheduler.step()

        # === チェックポイントの保存 ===
        # 定期的に、または最終エポックで、モデルとオプティマイザーの状態を保存
        if epoch % config.train.save_epochs == 0 or epoch == config.train.epochs or interrupted:
            save_checkpoint(model, optimizer, epoch, os.path.join(config.path.log_dir, f"checkpoint_{epoch}.pth"))

        # Ctrl+Cで中断された場合は、チェックポイント保存後にループを抜ける
        if interrupted:
            break

        # === 検証 ===
        # テストデータで性能を評価し、サンプルテキストを生成
        validate(model, test_loader, test_writer, tokenizer, epoch=epoch)

    # === 学習終了後の後処理 ===
    # TensorBoardのWriterを閉じる
    train_writer.close()
    test_writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, default="config/simple_stories.yaml")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (mixed precision) for debugging")
    args = parser.parse_args()

    # YAMLファイルから学習設定（バッチサイズ、学習率など）を読み込む
    config = SLMConfig.load(args.config)

    # 実験の再現性を確保するため、乱数シードを固定
    torch.manual_seed(config.train.manual_seed)
    torch.cuda.manual_seed(config.train.manual_seed)

    # CUDAが使える場合はGPU、使えない場合はCPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルとトークナイザー(テキストをトークンID列に変換する処理)を初期化
    # (KVキャッシュは無効)
    model, tokenizer = get_model(config, enable_internal_cache=False)

    # モデルをGPU/CPUに転送
    model.to(device)
    # PyTorch 2.0のコンパイル機能で高速化できる場合もある、今回はあまり効果なし
    # model = torch.compile(model)

    # オプティマイザーの初期化
    # AdamW: Adamに weight decay（L2正則化）を追加したオプティマイザー
    # betas: Adamのモーメンタムパラメータ（通常は [0.9, 0.999]）
    # lr: 学習率（パラメータ更新の大きさを制御）
    # weight_decay: L2正則化の強さ（過学習を防ぐ）
    optimizer = torch.optim.AdamW(
        model.parameters(), betas=config.train.betas, lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )

    # 学習途中から再開する場合、最新のチェックポイントを読み込む
    model_dir = config.path.log_dir
    checkpoint_path = latest_checkpoint_path(model_dir, "checkpoint_*.pth")
    if checkpoint_path is not None:
        start_epoch = (
            load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
            )
            + 1
        )
    else:
        start_epoch = 1

    # データセットの準備

    # 各文章の末尾に<|endoftext|>トークンを追加してトークンIDに変換する関数
    def encode_story(xs):
        xs.update({"story": [tokenizer.encode(x + "<|endoftext|>", allowed_special={"<|endoftext|>"}) for x in xs["story"]]})
        return xs

    if config.dataset == "SimpleStories-Both":
        # SimpleStoriesデータセットのJAとENの両方を使用

        # 学習用データセット
        train_dataset = SimpleStoriesBothDataset(subset="train")
        train_dataset = train_dataset.map(encode_story, batched=True)

        # 検証用データセット
        test_dataset = SimpleStoriesBothDataset(subset="test")
        test_dataset = test_dataset.map(encode_story, batched=True)
    elif "SimpleStories" in config.dataset:
        # 言語単体版のデータセットを使用する場合
        train_dataset = datasets.load_dataset(config.dataset, split="train")
        train_dataset = train_dataset.map(encode_story, batched=True)
        test_dataset = datasets.load_dataset(config.dataset, split="test")
        test_dataset = test_dataset.map(encode_story, batched=True)
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")

    # データセットからミニバッチを作成してモデルに供給するDataLoaderを作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,  # 一度に処理するサンプル数(バッチサイズ)
        pin_memory=True,  # 使用するメモリを固定する、ちょっとだけ高速化する効果がある
        shuffle=True,  # エポックごとにデータをシャッフルするかのフラグ
        collate_fn=lambda x: dataset_collate(
            x, torch_convert=True, max_length=config.train.max_length
        ),  # バッチの作り方を指定
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        pin_memory=True,
        shuffle=False,  # 検証時はシャッフル不要
        collate_fn=lambda x: dataset_collate(x, torch_convert=True, max_length=config.train.max_length),
    )

    # 学習の実行
    train(
        config,
        model,
        optimizer,
        train_loader,
        test_loader,
        tokenizer,
        start_epoch,
        enable_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
