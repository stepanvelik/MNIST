"""Командный интерфейс для распознавания цифр MNIST."""

import argparse
import time

from digit_recognizer import KNNClassifier, load_digit_image, load_mnist


def evaluate(args: argparse.Namespace) -> None:
    train_images, train_labels = load_mnist("train", args.data_dir, args.train_samples)
    test_images, test_labels = load_mnist("test", args.data_dir, args.samples)

    classifier = KNNClassifier(k=args.k).fit(train_images, train_labels)
    started = time.perf_counter()
    predictions = classifier.predict(test_images)
    elapsed = time.perf_counter() - started
    accuracy = float((predictions == test_labels).mean()) * 100

    print(f"Обучающих изображений: {len(train_images)}")
    print(f"Проверено изображений: {len(test_images)}")
    print(f"Точность: {accuracy:.2f}%")
    print(f"Время распознавания: {elapsed:.2f} с")


def predict(args: argparse.Namespace) -> None:
    train_images, train_labels = load_mnist("train", args.data_dir, args.train_samples)
    image = load_digit_image(args.image)

    classifier = KNNClassifier(k=args.k).fit(train_images, train_labels)
    digit = int(classifier.predict(image)[0])
    print(f"Распознанная цифра: {digit}")


def draw(args: argparse.Namespace) -> None:
    print("Загрузка обучающих примеров MNIST...")
    train_images, train_labels = load_mnist("train", args.data_dir, args.train_samples)
    classifier = KNNClassifier(k=args.k).fit(train_images, train_labels)

    from digit_recognizer.gui import DigitDrawingApp

    DigitDrawingApp(classifier).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Распознавание цифр алгоритмом k-ближайших соседей")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--k", type=int, default=3, help="число соседей (по умолчанию: 3)")
        command.add_argument("--data-dir", default="data", help="папка для файлов MNIST")
        command.add_argument(
            "--train-samples",
            type=int,
            default=None,
            help="ограничить число обучающих примеров (для быстрого запуска)",
        )

    evaluate_parser = subparsers.add_parser("evaluate", help="проверить точность на тесте MNIST")
    evaluate_parser.add_argument("--samples", type=int, default=1000, help="число тестовых изображений")
    common_options(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate)

    predict_parser = subparsers.add_parser("predict", help="распознать цифру на картинке")
    predict_parser.add_argument("image", help="путь к PNG, JPG или другому изображению")
    common_options(predict_parser)
    predict_parser.set_defaults(handler=predict)

    draw_parser = subparsers.add_parser("draw", help="открыть окно для рисования цифры")
    common_options(draw_parser)
    draw_parser.set_defaults(handler=draw)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()
