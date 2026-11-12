const TAB_DRAFT = '3dfun-upload-draft';

export function validateReferenceFile(file, maxMB = 20) {
  if (!file?.size || !/\.(png|jpe?g|webp|bmp)$/i.test(file.name)) {
    throw new Error('Choose a PNG, JPEG, WEBP or BMP image.');
  }
  if (file.size > maxMB * 1024 ** 2) throw new Error(`Choose an image under ${maxMB} MB.`);
}

async function draftStore(mode, operation) {
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('3dfun-reference-drafts', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('images');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction('images', mode);
      const request = operation(transaction.objectStore('images'));
      transaction.oncomplete = () => resolve(request.result);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally { db.close(); }
}

export async function rememberReference(file) {
  const id = crypto.randomUUID();
  const previous = sessionStorage.getItem(TAB_DRAFT);
  await draftStore('readwrite', store => {
    if (previous) store.delete(previous);
    // Bound abandoned drafts to one hour. A cursor reads keys before retrieving file data.
    const expired = store.openKeyCursor(IDBKeyRange.upperBound(`${Date.now() - 3600000}:`));
    expired.onsuccess = () => {
      const cursor = expired.result;
      if (cursor) { store.delete(cursor.primaryKey); cursor.continue(); }
    };
    return store.put(file, `${Date.now()}:${id}`);
  }).then(key => sessionStorage.setItem(TAB_DRAFT, key));
  return id;
}

export async function pendingReference(id) {
  const key = sessionStorage.getItem(TAB_DRAFT);
  if (!key || key.split(':')[1] !== id) return null;
  if (Date.now() - Number(key.split(':')[0]) > 3600000) {
    await discardReference(id);
    return null;
  }
  return draftStore('readonly', store => store.get(key));
}

export async function discardReference(id) {
  const key = sessionStorage.getItem(TAB_DRAFT);
  if (!key || key.split(':')[1] !== id) return;
  await draftStore('readwrite', store => store.delete(key));
  sessionStorage.removeItem(TAB_DRAFT);
}
