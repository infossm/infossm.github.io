const $ = (x) => document.querySelector(x);
const $$ = (x) => document.createElement(x);

let mainString = 'adcabcdabaca';
let alreadySuffix = [-1, 7, 3, 9, -1, -1, -1, -1, -1, -1, -1, -1];
let animatingPromise = undefined;

const handle = {};

const getUrlParams = () => {
  const params = {};
  window.location.search.replace(/[?&]+([^=&]+)=([^&]*)/gi, (_, key, value) => {
    params[key] = value;
  });
  return params;
};

const shuffle = (l) => {
  const n = l.length;
  if (n <= 1) return;
  for (let i=0; i<n; ++i) {
    const j = Math.floor(Math.random() * (n - i)) + i;
    if (i != j) {
      [l[i], l[j]] = [l[j], l[i]]
    }
  }
  return l;
};

Array.prototype.stable_sort = function (key) {
  const self = this;
  const compare = key || ((x, y) => `${x}` < `${y}` ? -1 : `${x}` > `${y}` ? 1 : 0);
  self.forEach((v, i) => {
    self[i] = [v, i];
  });
  self.sort(([v, i], [w, j]) => {
    const res = compare(v, w);
    if (res !== 0) {
      return res;
    }
    return i - j;
  });
  self.forEach(([v], i) => {
    self[i] = v;
  });
  return self;
};

const makeLS = () => {
  const arr = new Array(mainString.length).fill('L');
  for (let i=mainString.length-2; i>=0; --i) {
    if (mainString[i] < mainString[i+1]) {
      arr[i] = 'S';
    } else if (mainString[i] === mainString[i+1]) {
      arr[i] = arr[i+1];
    }
  }
  return arr;
};

const makeStringTable = () => {
  const table = $$('table');
  const LSTable = makeLS();
  const tr = $$('tr');
  table.appendChild(tr);
  const tds = LSTable.map((v, i) => {
    const td = $$('td');
    td.innerText = mainString[i];
    td.style.backgroundColor = `#${v === 'L' ? '68c2ee' : 'f8c8c9'}`;
    tr.appendChild(td);
    return td;
  });
  handle.mainStringTableTds = tds;
  return table;
};

const fetchSAColor = () => {
  const LSTable = makeLS();
  const LSs = {};
  LSTable.forEach((v, i) => {
    const character = mainString[i];
    if (LSs[character] === undefined) {
      LSs[character] = [0, 0];
    }
    ++LSs[character][v === 'L' ? 0 : 1];
  });
  const color = new Array(LSTable.length).fill('L');
  let i = 0;
  const LSArray = Object.entries(LSs).sort(([key1, _], [key2, __]) => key1 < key2 ? -1 : key1 > key2 ? 1 : 0);
  LSArray.forEach(([key, value]) => {
    const [L, S] = value;
    i += L;
    for (let j=0; j<S; ++j) {
      color[i++] = 'S';
    }
  });
  return [LSArray, color];
};

const makeSuffixArray = () => {
  const [LSArray, color] = fetchSAColor();

  const table = $$('table');
  const thead = $$('tr');
  const tr = $$('tr');
  table.appendChild(thead);
  table.appendChild(tr);
  LSArray.forEach(([character, [L, S]]) => {
    const td = $$('td');
    const div = $$('div');
    td.setAttribute('colspan', `${L + S}`);
    td.classList.add('bold-left');
    td.classList.add('bold-right');
    td.classList.add('heading');

    const [lineLeft, lineRight] = [0, 0].map(() => {
      const line = $$('div');
      line.classList.add('heading-line');
      line.style.width = `${((L + S) * 50 - 30) / 2 - 1.5}px`;
      return line;
    });
    const divText = $$('div');
    divText.innerText = character;
    div.appendChild(lineLeft);
    div.appendChild(divText);
    div.appendChild(lineRight);

    td.appendChild(div);
    thead.appendChild(td);
  });
  const mainStringSortedTable = [...mainString].sort();
  const tds = alreadySuffix.map((x, i) => {
    const td = $$('td');
    if (x >= 0) {
      td.innerText = `${x}`;
    } else {
      td.classList.add('clear-all');
    }
    td.style.backgroundColor = `#${color[i] === 'L' ? '68c2ee' : 'f8c8c9'}`;
    if (i === 0 || (mainStringSortedTable[i] !== mainStringSortedTable[i-1])) {
      td.classList.add('bold-left');
    }
    if (i === alreadySuffix.length - 1 || (mainStringSortedTable[i] !== mainStringSortedTable[i+1])) {
      td.classList.add('bold-right');
    }
    tr.appendChild(td);
    return td;
  });
  handle.SATableTds = tds;
  return table;
};

const init = () => {
  const urlParams = getUrlParams();
  if (urlParams.str !== undefined) {
    mainString = urlParams.str;
  }
  if (urlParams.random === '1') {
    alreadySuffix = new Array(mainString.length).fill(-1);
    const color = fetchSAColor()[1].map((v, i) => [v, i]).filter(([v]) => v === 'S').map(([_, i]) => i);
    const suffixSs = makeLS().map((v, i) => [v, i]).filter(([v]) => v === 'S').map(([_, i]) => [mainString[i], i]);
    shuffle(suffixSs).stable_sort(([v], [w]) => v < w ? -1 : v > w ? 1 : 0).forEach(([_, idx], i) => {
      alreadySuffix[color[i]] = idx;
    });
  } else {
    alreadySuffix = [-1, 7, 3, 9, -1, -1, -1, -1, -1, -1, -1, -1];
  }
  const suffixArrayMessage = $$('div');
  suffixArrayMessage.innerText = 'Suffix Array';
  suffixArrayMessage.classList.add('message');
  const suffixArrayInfo = $$('div');
  suffixArrayInfo.innerText = 'Click to Play';
  suffixArrayInfo.classList.add('message-info');

  [makeStringTable(), suffixArrayMessage, makeSuffixArray(), suffixArrayInfo].forEach((elem) => {
    $('div.wrapper').appendChild(elem);
  });
  $('body').addEventListener('mousedown', handleClick);
  handle.infoDiv = suffixArrayInfo;
};

const asyncResolve = (isLong) => new Promise((resolve) => animatingPromise = setTimeout(resolve, isLong ? 1000 : 300));

const forwardPointer = () => {
  const pointer = Object.entries([...mainString].reduce((prev, curr) => {
    if (prev[curr] === undefined) {
      prev[curr] = 0;
    }
    ++prev[curr];
    return prev;
  }, {})).sort(([key1, _], [key2, __]) => key1 < key2 ? -1 : key1 > key2 ? 1 : 0).reduce((prev, [_, curr]) => {
    prev.push([_, prev[prev.length - 1][1] + curr]);
    return prev;
  }, [[null, 0]]);
  const res = pointer.map((_, i) => [i + 1 < pointer.length ? pointer[i+1][0] : null, pointer[i][1]]);
  res.pop();
  return Object.fromEntries(res);
};

const backwardPointer = () => {
  const pointer = Object.entries([...mainString].reduce((prev, curr) => {
    if (prev[curr] === undefined) {
      prev[curr] = 0;
    }
    ++prev[curr];
    return prev;
  }, {})).sort(([key1, _], [key2, __]) => key1 < key2 ? -1 : key1 > key2 ? 1 : 0).reduce((prev, [_, curr]) => {
    prev.push([_, prev[prev.length - 1][1] + curr]);
    return prev;
  }, [[null, 0]]);
  pointer.splice(0, 1);
  return Object.fromEntries(pointer);
};

let animationTimer = 0;
const asyncAnimate = async () => {
  const fpointer = forwardPointer();
  const bpointer = backwardPointer();
  const n = mainString.length;
  const { mainStringTableTds, SATableTds } = handle;
  const LS = makeLS();
  // step 1
  handle.infoDiv.innerText = 'Step 1';
  await asyncResolve(true);
  {
    mainStringTableTds[n-1].classList.add('color-red');
    await asyncResolve(true);
    const c = mainString[n-1];
    alreadySuffix[fpointer[c]] = n - 1;
    SATableTds[fpointer[c]].innerText = `${n - 1}`;
    SATableTds[fpointer[c]].classList.remove('clear-all');
    SATableTds[fpointer[c]].classList.add('color-red');
    await asyncResolve(true);
    mainStringTableTds[n-1].classList.remove('color-red');
    SATableTds[fpointer[c]].classList.remove('color-red');
    ++fpointer[c];
    await asyncResolve(true);
  }
  await asyncResolve(true);
  // step 2
  handle.infoDiv.innerText = 'Step 2';
  await asyncResolve(true);
  {
    for (let i=0; i<n; ++i) {
      const idx = alreadySuffix[i];
      if (idx >= 0) {
        SATableTds[i].classList.add('color-red');
        await asyncResolve(true);
        if (idx > 0) {
          mainStringTableTds[idx - 1].classList.add('color-red');
          await asyncResolve(true);
          if (LS[idx - 1] === 'L') {
            let c = mainString[idx - 1];
            alreadySuffix[fpointer[c]] = idx - 1;
            SATableTds[fpointer[c]].innerText = `${idx - 1}`;
            SATableTds[fpointer[c]].classList.remove('clear-all');
            SATableTds[fpointer[c]].classList.add('color-red');
            await asyncResolve(true);
            SATableTds[fpointer[c]].classList.remove('color-red');
            ++fpointer[c];
          }
          mainStringTableTds[idx - 1].classList.remove('color-red');
        }
        SATableTds[i].classList.remove('color-red');
        await asyncResolve(true);
      }
    }
  }
  await asyncResolve(true);
  // step 3
  handle.infoDiv.innerText = 'Step 3';
  await asyncResolve(true);
  {
    for (let i=0; i<n; ++i) {
      if (alreadySuffix[i] >= 0 && LS[alreadySuffix[i]] === 'S') {
        SATableTds[i].classList.add('color-red');
      }
    }
    await asyncResolve(true);
    for (let i=0; i<n; ++i) {
      if (alreadySuffix[i] >= 0 && LS[alreadySuffix[i]] === 'S') {
        SATableTds[i].classList.remove('color-red');
        SATableTds[i].classList.add('clear-all');
        alreadySuffix[i] = -1;
      }
    }
    await asyncResolve(true);
  }
  await asyncResolve(true);
  // step 4
  handle.infoDiv.innerText = 'Step 4';
  await asyncResolve(true);
  {
    for (let i=n-1; i>=0; --i) {
      const idx = alreadySuffix[i];
      if (idx >= 0) {
        SATableTds[i].classList.add('color-red');
        await asyncResolve(true);
        if (idx > 0) {
          mainStringTableTds[idx - 1].classList.add('color-red');
          await asyncResolve(true);
          if (LS[idx - 1] === 'S') {
            let c = mainString[idx - 1];
            --bpointer[c];
            alreadySuffix[bpointer[c]] = idx - 1;
            SATableTds[bpointer[c]].innerText = `${idx - 1}`;
            SATableTds[bpointer[c]].classList.remove('clear-all');
            SATableTds[bpointer[c]].classList.add('color-red');
            await asyncResolve(true);
            SATableTds[bpointer[c]].classList.remove('color-red');
          }
          mainStringTableTds[idx - 1].classList.remove('color-red');
        }
        SATableTds[i].classList.remove('color-red');
        await asyncResolve(true);
      }
    }
  }
  await asyncResolve(true);
  handle.infoDiv.innerText = 'Click to Replay';
  animationTimer = 2;
  animatingPromise = undefined;
};

const handleClick = () => {
  if (animationTimer === 0) {
    animationTimer = 1;
    handle.infoDiv.innerText = '';
    animatingPromise = setTimeout(asyncAnimate, 500);
  } else if (animationTimer === 1) {
    if (getUrlParams().random === '1') {
      const wrapper = $('div.wrapper');
      while (wrapper.firstChild) {
        wrapper.removeChild(wrapper.lastChild);
      }
      init();
      animationTimer = 0;
      if (animatingPromise !== undefined) {
        clearTimeout(animatingPromise);
        animatingPromise = undefined;
      }
    }
  } else if (animationTimer === 2) {
    const wrapper = $('div.wrapper');
    while (wrapper.firstChild) {
      wrapper.removeChild(wrapper.lastChild);
    }
    init();
    animationTimer = 0;
  }
};
