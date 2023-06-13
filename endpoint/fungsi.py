
import numpy as np
import pandas as pd
def toLowercase(df):
  return [str(lower).lower() for lower in df]

def toList(df):
    return [i.split(', ') for i in df]

def encode_usia(usia,jenis_usia, jumlah_jenis_usia, user = False):
    # Membuat vektor nol dengan panjang yang sama dengan jumlah usia
    encoded_usia = [0] * jumlah_jenis_usia

    for i in usia:  
      # Menentukan indeks usia yang sesuai
      
      usia_index = jenis_usia.index(i)
      
      # Mengatur nilai 1 pada indeks usia yang sesuai
      encoded_usia[usia_index] = 1

    if user:
      encoded_usia[0] = 1
    
    return encoded_usia

def encode_category(category,jenis_kategori, jumlah_kategori):
    # Membuat vektor nol dengan panjang yang sama dengan jumlah kategori
    encoded_category = [0] * jumlah_kategori

    for i in category:  
      # Menentukan indeks kategori yang sesuai
      category_index = jenis_kategori.index(i)
      
      # Mengatur nilai 1 pada indeks kategori yang sesuai
      encoded_category[category_index] = 1
    
    
    return encoded_category

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

def convert_usia_user(df):
    if df.values[0] >= 15 and df.values[0] < 18:
        df = ['18-25']
    elif df.values[0] >= 18 and df.values[0] < 26:
        df = ['18-25']
    elif df.values[0] >= 26 and df.values[0] < 36:
        df = ['26-35']
    elif df.values[0] >= 36 and df.values[0] < 46:
        df = ['36-45']
    elif df.values[0] >= 46 and df.values[0] < 56:
        df = ['46-55']
    else:
        df = ['umum']
        
    return df
    
# Fungsi rekomendasi kuesioner
def pred_kuesioner(y_p, item, items, maxcount=10):
    count = 0
    df_pred = pd.DataFrame(columns=["kuesioner_id", "ratarata_rating", "judul","deskripsi", "rentang_umur", "kategori", "link"])

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        kuesioner_id = item[i, 0].astype(int) - 1
        new_row = pd.Series({
            "kuesioner_id": item[i, 0].astype(int), 
            "ratarata_rating": np.around(item[i, 2].astype(float), 1), 
            "judul": items[kuesioner_id,1],
            "deskripsi": items[kuesioner_id,2],
            "rentang_umur": items[kuesioner_id, 3], 
            "kategori": items[kuesioner_id,4],
            "link": items[kuesioner_id,6]
        })
        df_pred = pd.concat([df_pred, new_row.to_frame().T], ignore_index=True)
    return df_pred