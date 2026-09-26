#!/usr/bin/env python3
"""
Generates synthetic train and test datasets mimicking the exact structure,
column headers, and noise characteristics described in the problem statement.
"""
from pathlib import Path
import random

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Sample base entities
ENTITIES_TRAIN = [
    ("Acme Industrial Corp", "104 Main Street, Austin, TX 78701", "US"),
    ("Infosys Limited", "Electronics City, Hosur Road, Bangalore 560100", "India"),
    ("Global Tech Solutions Pvt Ltd", "Tower 4, Cyber City, Gurgaon, Haryana 122002", "India"),
    ("Metro Logistics LLC", "500 Harbor Blvd, Suite 200, Seattle, WA 98104", "US"),
    ("Apex Healthcare Inc", "12 Medical Center Dr, Boston, MA 02115", "US"),
    ("Tata Consultancy Services", "Banyan Park, Suren Road, Andheri East, Mumbai 400093", "India"),
    ("Blue Horizon Media Group", "742 Evergreen Terrace, Springfield, OR 97477", "US"),
    ("Reliance Retail Ventures", "Maker Chambers IV, Nariman Point, Mumbai 400021", "India"),
    ("Pinnacle Financial Partners", "150 3rd Avenue South, Nashville, TN 37201", "US"),
    ("Zenith Software Private Ltd", "Marathahalli Outer Ring Rd, Bangalore 560037", "India"),
]

ENTITIES_TEST = [
    ("Beacon Health System", "615 N Michigan St, South Bend, IN 46601", "US"),
    ("Larsen & Toubro Ltd", "L&T House, Ballard Estate, Mumbai 400001", "India"),
    ("Société Générale SA", "29 Boulevard Haussmann, 75009 Paris", "France"),
    ("Vanguard Asset Management", "100 Vanguard Blvd, Malvern, PA 19355", "US"),
    ("Wipro Enterprises Pvt Ltd", "Sarjapur Road, Doddakannelli, Bangalore 560035", "India"),
    ("Carrefour Hypermarket", "93 Avenue de Paris, 91300 Massy", "France"),
    ("HCL Technologies Ltd", "Technology Hub, Plot 3A, Sector 126, Noida 201304", "India"),
    ("Summit Ridge Energy LLC", "1401 Wilson Blvd, Suite 800, Arlington, VA 22209", "US"),
]


def perturb_name(name: str) -> str:
    replacements = {
        "Corp": "Corporation",
        "Corporation": "Corp",
        "Limited": "Ltd",
        "Ltd": "Limited",
        "Pvt Ltd": "Private Limited",
        "Private Ltd": "Pvt Ltd",
        "LLC": "Co.",
        "&": "and",
        "SA": "S.A.",
    }
    for k, v in replacements.items():
        if k in name:
            name = name.replace(k, v)
            break
    return name


def perturb_address(addr: str) -> str:
    replacements = {
        "Street": "St.",
        "Road": "Rd.",
        "Boulevard": "Blvd",
        "Avenue": "Ave",
        "Drive": "Dr",
        "Suite": "Ste",
    }
    for k, v in replacements.items():
        if k in addr:
            addr = addr.replace(k, v)
            break
    return addr


def generate_data():
    # 1. TRAIN SET
    train_s1, train_s2, train_s3 = [], [], []
    train_gt = []

    s2_idx, s3_idx = 1, 1
    for i, (name, addr, country) in enumerate(ENTITIES_TRAIN, start=1):
        s1_id = f"S1-{i:05d}"
        train_s1.append((s1_id, name, addr, country))

        matched = []
        # Entity 1 & 2 have matches in both S2 and S3
        if i in [1, 2, 6, 8]:
            s2_id = f"S2-{s2_idx:05d}"
            s2_idx += 1
            train_s2.append((s2_id, perturb_name(name), perturb_address(addr), country))
            matched.append(s2_id)

            s3_id = f"S3-{s3_idx:05d}"
            s3_idx += 1
            train_s3.append((s3_id, name.upper(), addr, country))
            matched.append(s3_id)
        # Entity 3 & 4 have match only in S2
        elif i in [3, 4, 9]:
            s2_id = f"S2-{s2_idx:05d}"
            s2_idx += 1
            train_s2.append((s2_id, perturb_name(name), addr, country))
            matched.append(s2_id)
        # Entity 5 has match only in S3
        elif i in [5, 10]:
            s3_id = f"S3-{s3_idx:05d}"
            s3_idx += 1
            train_s3.append((s3_id, name, perturb_address(addr), country))
            matched.append(s3_id)
        # Entity 7 is a singleton (no matches)
        else:
            pass

        train_gt.append((s1_id, ",".join(matched)))

    # Add random distractor records into S2 and S3
    train_s2.append((f"S2-{s2_idx:05d}", "Random Unrelated Shop LLC", "123 Nowhere Lane, Denver, CO", "US"))
    s2_idx += 1
    train_s3.append((f"S3-{s3_idx:05d}", "Unrelated Cyber Enterprise", "MG Road, Pune, Maharashtra", "India"))

    # Write Train Files
    for filename, rows in [
        ("train_source1.tsv", train_s1),
        ("train_source2.tsv", train_s2),
        ("train_source3.tsv", train_s3),
    ]:
        with open(TRAIN_DIR / filename, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                f.write("\t".join(r) + "\n")

    with open(TRAIN_DIR / "train_ground_truth.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in train_gt:
            f.write(f"{r[0]}\t{r[1]}\n")

    # 2. TEST SET
    test_s1, test_s2, test_s3 = [], [], []
    t_s2_idx, t_s3_idx = 1, 1

    for i, (name, addr, country) in enumerate(ENTITIES_TEST, start=1):
        s1_id = f"S1-{i:05d}"
        test_s1.append((s1_id, name, addr, country))

        if i in [1, 3, 5]:  # matches in S2
            s2_id = f"S2-{t_s2_idx:05d}"
            t_s2_idx += 1
            test_s2.append((s2_id, perturb_name(name), perturb_address(addr), country))

        if i in [2, 3, 6]:  # matches in S3
            s3_id = f"S3-{t_s3_idx:05d}"
            t_s3_idx += 1
            test_s3.append((s3_id, name.lower(), addr, country))

    # Add distractors to test
    test_s2.append((f"S2-{t_s2_idx:05d}", "French Bakery Distractor", "10 Rue de la Paix, Paris", "France"))
    test_s3.append((f"S3-{t_s3_idx:05d}", "US Electronics Surplus", "99 Elm Street, Dallas, TX", "US"))

    for filename, rows in [
        ("test_source1.tsv", test_s1),
        ("test_source2.tsv", test_s2),
        ("test_source3.tsv", test_s3),
    ]:
        with open(TEST_DIR / filename, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                f.write("\t".join(r) + "\n")

    print(f"Synthetic dataset generated in {DATASET_DIR}")


if __name__ == "__main__":
    generate_data()
