function timeToMinutes(t) {
  const [h, m] = t.split(":").map(Number);
  return h * 60 + m;
}

export function create_speaker_table(speakers, container, bound, use_date, use_youtube, sort_by_name, sort_by_date) {

  // Select container
  container.style.overflowX = "auto";
  container.style.width = "100%";

  // Create table
  const table = document.createElement("table");
  table.className = "border-collapse text-left";
  table.style.tableLayout = "fixed";

  // Create colgroup (widths now work reliably)
  const colgroup = document.createElement("colgroup");

  const widths = [50, 200, 200, 500];
  if( use_date ){
    widths.push(120);
  }
  if( use_youtube )
  {
    widths.push(100);
  }
  
  widths.forEach(w => {
    const col = document.createElement("col");
    col.style.width = w + "px";
    colgroup.appendChild(col);
  });
  table.appendChild(colgroup);

  // ----------------------
  // Build the table header
  // ----------------------
  const thead = document.createElement("thead");
  thead.className = "border-b font-semibold";

  const headerRow = document.createElement("tr");
  const headerNames = ["", "Name", "Affiliation", "Title"];
  if( use_date ){
    headerNames.push("Date");
  }
  if( use_youtube ){
    headerNames.push("");
  }
  headerNames.forEach(text => {
    const th = document.createElement("th");
    th.className = "p-2";
    th.textContent = text;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  // ----------------------
  // Select ONLY proper speakers
  // ----------------------
  const proper_speakers = speakers.filter(item => item.improper === undefined);
  
  if ( sort_by_name ) {
    proper_speakers.sort((a,b) => a.lastname.localeCompare(b.lastname));
  }
  else if ( sort_by_date ) {
    proper_speakers.sort((a,b) => a.sortdate.localeCompare(b.sortdate));
  }
  const selected_speakers = proper_speakers.slice(0, bound);

  // ----------------------
  // Build table body
  // ----------------------
  const tbody = document.createElement("tbody");

  const modal = document.getElementById("abstract-modal");
  const modalTitle = document.getElementById("abstract-title");
  const modalText = document.getElementById("abstract-text");
  const modalClose = document.getElementById("close-modal");
  const modalContent = document.getElementById("abstract-content");

  modalClose.onclick = () => modal.style.display = "none";
  modal.onclick = (e) => {
    if (e.target === modal) modal.style.display = "none";
  };

  selected_speakers.forEach(speaker => {
    const row = document.createElement("tr");
    row.className = "border-b";

    // --- Picture cell ---
    const pictureCell = document.createElement("td");
    pictureCell.className = "p-2";

    if( speaker.picture ) {
      const img = document.createElement("img");
      img.src = speaker.picture || "img/default.png";
      img.alt = speaker.name;

      img.style.width = "90px";
      img.style.height = "90px";
      img.style.borderRadius = "50%";
      img.style.objectFit = "cover";
      img.style.display = "block";

      pictureCell.appendChild(img);
    }
    row.appendChild(pictureCell);

    // --- Name ---
    const nameCell = document.createElement("td");
    nameCell.className = "p-2";
    nameCell.textContent = `${speaker.firstname} ${speaker.lastname}`;
    row.appendChild(nameCell);

    // --- Affiliation (clickable if website exists) ---
    const affCell = document.createElement("td");
    affCell.className = "p-2";

    if (speaker.website && speaker.website.trim() !== "") {
      const link = document.createElement("a");
      link.href = speaker.website;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.style.color = "#1a0dab";
      link.textContent = speaker.affiliation ?? "";
      affCell.appendChild(link);
    } else {
      affCell.textContent = speaker.affiliation ?? "";
    }

    row.appendChild(affCell);

    // --- Title ---
    const titleCell = document.createElement("td");
    titleCell.className = "p-2 text-blue-700 underline cursor-pointer";
    titleCell.textContent = speaker.title;
    titleCell.style = "min-width: 250px;";

    // add icon to show that this is a link
    if ( speaker.abstracttext && speaker.abstracttext.trim() != "") {
    const absLink = document.createElement("span");
    absLink.innerHTML = `
  <svg xmlns="http://www.w3.org/2000/svg"
       width="16" height="16" viewBox="0 0 24 24"
       fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
    <polyline points="15 3 21 3 21 9"/>
    <line x1="10" y1="14" x2="21" y2="3"/>
  </svg>
`;

    absLink.style.marginLeft = "6px";
    absLink.style.cursor = "pointer";
    absLink.style.verticalAlign = "middle";

    titleCell.appendChild(absLink);

    }


    // When clicked → show popup
    titleCell.addEventListener("click", () => {
      modalTitle.textContent = speaker.title;
      modalText.textContent = speaker.abstracttext ?? "No abstract available.";
      modalText.innerHTML = speaker.abstracttext.replace(/\n/g, "<br>");
      modal.style.display = "flex";
    });

    row.appendChild(titleCell);
    ``

    // --- Date ---
    if( use_date ){
      const dateCell = document.createElement("td");
      dateCell.className = "p-2";
      dateCell.textContent = speaker.date ?? "";
      row.appendChild(dateCell);
    }

    // --- YouTube icon ---
    if( use_youtube )
    {
      const ytCell = document.createElement("td");
      ytCell.className = "p-2";

      if (speaker.youtube && speaker.youtube.trim() !== "") {
	const ytLink = document.createElement("a");
	ytLink.href = speaker.youtube;
	ytLink.target = "_blank";
	ytLink.rel = "noopener noreferrer";

	const ytIcon = document.createElement("img");
	ytIcon.src = "img/youtube.png";  // or youtube.svg
	ytIcon.alt = "YouTube";
	ytIcon.style.width = "32px";
	ytIcon.style.height = "32px";
	ytIcon.style.cursor = "pointer";

	ytLink.appendChild(ytIcon);
	ytCell.appendChild(ytLink);
      }

      row.appendChild(ytCell);
    }

    // Add row to tbody
    tbody.appendChild(row);
  });

  // Add tbody to table
  table.appendChild(tbody);

  // Add table to page
  container.appendChild(table);
}

export function create_timetable(speakers, container, day) {

  // Select container
  container.style.overflowX = "auto";
  container.style.width = "100%";

  // Create table
  const table = document.createElement("table");
  table.className = "border-collapse text-left";
  table.style.tableLayout = "fixed";

  // Create colgroup (widths now work reliably)
  const colgroup = document.createElement("colgroup");

  const widths = [100, 200, 600];
  
  widths.forEach(w => {
    const col = document.createElement("col");
    col.style.width = w + "px";
    colgroup.appendChild(col);
  });
  table.appendChild(colgroup);

  // ----------------------
  // Build the table header
  // ----------------------
  const thead = document.createElement("thead");
  thead.className = "border-b font-semibold";

  const headerRow = document.createElement("tr");
  const headerNames = ["Time", "Name", "Title"];
  headerNames.forEach(text => {
    const th = document.createElement("th");
    th.className = "p-2";
    th.textContent = text;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  // ----------------------
  // Build table body
  // ----------------------
  const tbody = document.createElement("tbody");

  const modal = document.getElementById("abstract-modal");
  const modalTitle = document.getElementById("abstract-title");
  const modalText = document.getElementById("abstract-text");
  const modalClose = document.getElementById("close-modal");
  const modalContent = document.getElementById("abstract-content");

  modalClose.onclick = () => modal.style.display = "none";
  modal.onclick = (e) => {
    if (e.target === modal) modal.style.display = "none";
  };

  const selected_speakers = speakers.filter(item => item.sortdate === day)
  console.log(speakers);
  console.log(selected_speakers);
  selected_speakers.sort((a,b) => {
    const startA = a.time.split("-")[0];
    const startB = b.time.split("-")[0];
    return timeToMinutes(startA) - timeToMinutes(startB);
  });

  selected_speakers.forEach(speaker => {
    const row = document.createElement("tr");
    row.className = "border-b";

    // --- Time cell ---
    const timeCell = document.createElement("td");
    timeCell.className = "p-2";
    timeCell.textContent = speaker.time;
    row.appendChild(timeCell);

    // --- Name ---
    const nameCell = document.createElement("td");
    nameCell.className = "p-2";
    nameCell.textContent = `${speaker.firstname} ${speaker.lastname}`;
    row.appendChild(nameCell);

    // --- Title ---
    const titleCell = document.createElement("td");
    titleCell.className = "p-2 text-blue-700 underline cursor-pointer";
    if ( speaker.title ) {
      titleCell.textContent = speaker.title;
      titleCell.style = "min-width: 250px;";

      // add icon to show that this is a link
      if ( speaker.abstracttext && speaker.abstracttext.trim() != "") {
	const absLink = document.createElement("span");
	absLink.innerHTML = `
  <svg xmlns="http://www.w3.org/2000/svg"
       width="16" height="16" viewBox="0 0 24 24"
       fill="none" stroke="currentColor"
       stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
    <polyline points="15 3 21 3 21 9"/>
    <line x1="10" y1="14" x2="21" y2="3"/>
  </svg>
`;

	absLink.style.marginLeft = "6px";
	absLink.style.cursor = "pointer";
	absLink.style.verticalAlign = "middle";

	titleCell.appendChild(absLink);
      }

      // When clicked → show popup
      titleCell.addEventListener("click", () => {
	modalTitle.textContent = speaker.title;
	modalText.textContent = speaker.abstracttext ?? "No abstract available.";
	modalText.innerHTML = speaker.abstracttext.replace(/\n/g, "<br>");
	modal.style.display = "flex";
      });
    }

    row.appendChild(titleCell);
    ``

    // Add row to tbody
    tbody.appendChild(row);
  });

  // Add tbody to table
  table.appendChild(tbody);

  // Add table to page
  container.appendChild(table);
}

export function create_confirmed_speaker_table(speakers, container) {
  create_speaker_table(speakers, container, speakers.length, false, false, true, false);
}
