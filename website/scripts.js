const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const indexLabel = document.getElementById('indexLabel');

// 画像ファイル名の配列
const imagePaths = Array.from({length: 10}, (_, i) => `images/img${i}.jpg`);

// Imageオブジェクトをまとめて読み込む
const images = imagePaths.map(path => {
  const img = new Image();
  img.src = path;
  return img;
});

// 全画像読み込み完了を待つユーティリティ
Promise.all(images.map(img => 
  new Promise(resolve => img.onload = resolve)
)).then(() => {
  // 初期表示
  drawImage(+slider.value);
});

// スライダー操作で画像切り替え
slider.addEventListener('input', () => {
  const idx = +slider.value;
  indexLabel.textContent = idx;
  drawImage(idx);
});

function drawImage(idx) {
  const img = images[idx];
  // 画面に合わせて描画
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}
