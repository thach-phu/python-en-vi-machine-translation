import argparse
import json
import os
import sys
import sacrebleu


def evaluate_translation(preds_file: str, refs_files: list, verbose: bool = True) -> dict:
    """
    Hàm tính điểm SacreBLEU và chrF++ giữa file dự đoán và (các) file tham chiếu.

    :param preds_file: Đường dẫn đến file chứa các câu dịch của mô hình (mỗi câu 1 dòng)
    :param refs_files: Danh sách đường dẫn đến (các) file chứa câu dịch chuẩn (mỗi câu 1 dòng)
    :param verbose: In kết quả đẹp ra màn hình console
    :return: Dictionary chứa điểm BLEU, chrF và signature
    """
    # 1. Kiểm tra file tồn tại
    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"❌ Không tìm thấy file dự đoán: {preds_file}")

    for ref in refs_files:
        if not os.path.exists(ref):
            raise FileNotFoundError(f"❌ Không tìm thấy file tham chiếu: {ref}")

    # 2. Đọc file dự đoán
    with open(preds_file, "r", encoding="utf-8") as f:
        sys_preds = [line.strip() for line in f]

    # 3. Đọc (các) file tham chiếu
    # sacrebleu yêu cầu refs dạng list of lists: [[ref1_line1, ref1_line2], [ref2_line1, ref2_line2]]
    refs = []
    for ref_file in refs_files:
        with open(ref_file, "r", encoding="utf-8") as f:
            refs.append([line.strip() for line in f])

    # 4. Kiểm tra độ dài
    num_preds = len(sys_preds)
    for i, ref_list in enumerate(refs):
        if len(ref_list) != num_preds:
            raise ValueError(
                f"⚠️ Lỗi chênh lệch số dòng! File dự đoán có {num_preds} dòng, "
                f"nhưng file tham chiếu '{refs_files[i]}' có {len(ref_list)} dòng."
            )

    # 5. Tính toán các chỉ số
    # SacreBLEU score
    bleu = sacrebleu.corpus_bleu(sys_preds, refs)

    # chrF++ score (mặc định beta=2, char_order=6, word_order=2)
    chrf = sacrebleu.corpus_chrf(sys_preds, refs)

    results = {
        "total_sentences": num_preds,
        "bleu_score": round(bleu.score, 2),
        "bleu_signature": str(bleu.get_signature()),
        "chrf_score": round(chrf.score, 2),
        "chrf_signature": str(chrf.get_signature()),
    }

    # 6. Hiển thị kết quả
    if verbose:
        print("\n==============================================")
        print("          KẾT QUẢ ĐÁNH GIÁ (EVALUATION)       ")
        print("==============================================")
        print(f"📊 Số lượng câu đánh giá: {num_preds:,}")
        print(f"🔹 BLEU score   : {results['bleu_score']:.2f}")
        print(f"   Signature    : {results['bleu_signature']}")
        print(f"🔹 chrF++ score  : {results['chrf_score']:.2f}")
        print(f"   Signature    : {results['chrf_signature']}")
        print("==============================================\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Script đánh giá mô hình dịch máy bằng SacreBLEU và chrF++"
    )

    parser.add_argument(
        "--preds", "-p",
        type=str,
        required=True,
        help="Đường dẫn đến file câu dự đoán (predictions/hypotheses)"
    )

    parser.add_argument(
        "--refs", "-r",
        type=str,
        nargs="+",
        required=True,
        help="Đường dẫn đến (các) file câu dịch chuẩn (references)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="[Tùy chọn] Đường dẫn file JSON để lưu kết quả đánh giá"
    )

    args = parser.parse_args()

    try:
        results = evaluate_translation(args.preds, args.refs)

        # Lưu file kết quả JSON nếu người dùng truyền tham số --output
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"✅ Đã lưu kết quả đánh giá vào: {args.output}")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()