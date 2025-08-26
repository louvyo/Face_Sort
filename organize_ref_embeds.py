import os
import shutil

REF_EMBEDS_DIR = r'ref_embeds'

def should_delete(filename):
    # Contoh: hapus file yang mengandung 'old' atau 'duplicate'
    return 'old' in filename or 'duplicate' in filename

def get_person_name(filename):
    # Asumsi: nama orang adalah bagian awal sebelum underscore
    # Contoh: person1_embed1.pkl -> person1
    return filename.split('_')[0]

def organize_embeddings():
    for fname in os.listdir(REF_EMBEDS_DIR):
        fpath = os.path.join(REF_EMBEDS_DIR, fname)
        if os.path.isfile(fpath):
            if should_delete(fname):
                os.remove(fpath)
                print(f'Deleted: {fname}')
            else:
                person = get_person_name(fname)
                person_dir = os.path.join(REF_EMBEDS_DIR, person)
                os.makedirs(person_dir, exist_ok=True)
                shutil.move(fpath, os.path.join(person_dir, fname))
                print(f'Moved: {fname} -> {person}/')

if __name__ == '__main__':
    organize_embeddings()
