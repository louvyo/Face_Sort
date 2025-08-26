import os
import shutil

BASE_DIR = r'd:\face_sortir_web'
REF_EMBEDS_DIR = os.path.join(BASE_DIR, 'ref_embeds')

def get_person_name(filename):
    # Asumsi: nama orang adalah bagian awal sebelum underscore
    return filename.split('_')[0]

def move_ref_files():
    for root, dirs, files in os.walk(BASE_DIR):
        # Lewati folder ref_embeds
        if REF_EMBEDS_DIR in root:
            continue
        for fname in files:
            # Asumsi file embedding berekstensi .pkl
            if fname.endswith('.pkl'):
                src_path = os.path.join(root, fname)
                person = get_person_name(fname)
                person_dir = os.path.join(REF_EMBEDS_DIR, person)
                os.makedirs(person_dir, exist_ok=True)
                dst_path = os.path.join(person_dir, fname)
                shutil.move(src_path, dst_path)
                print(f'Moved: {src_path} -> {dst_path}')

if __name__ == '__main__':
    move_ref_files()
