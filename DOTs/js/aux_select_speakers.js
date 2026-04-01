// import { past_speakers } from './past_talks.js';

export function create_speaker_table(speakers, container, bound) {

  // Select container
  container.style.overflowX = "auto";
  container.style.width = "100%";

  // Create table
  const table = document.createElement("table");
  table.className = "border-collapse text-left";
  table.style.tableLayout = "fixed";
  // table.style.width = "1500px"; // ensure enough width for colgroup

  // Create colgroup (widths now work reliably)
  const colgroup = document.createElement("colgroup");

  const widths = [50, 100, 70, 400, 120, 100];
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
  ["", "Name", "Affiliation", "Title", "Date", ""].forEach(text => {
    const th = document.createElement("th");
    th.className = "p-2";
    th.textContent = text;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);

  // ----------------------
  // Select ONLY recent speakers
  // ----------------------
  const selected_speakers = speakers.slice(0, bound);

  // ----------------------
  // Build table body
  // ----------------------
  const tbody = document.createElement("tbody");

  const modal = document.getElementById("abstract-modal");
  const modalTitle = document.getElementById("abstract-title");
  const modalText = document.getElementById("abstract-text");
  const modalClose = document.getElementById("close-modal");

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

    const img = document.createElement("img");
    img.src = speaker.picture || "img/default.png";
    img.alt = speaker.name;

    img.style.width = "90px";
    img.style.height = "90px";
    img.style.borderRadius = "50%";
    img.style.objectFit = "cover";
    img.style.display = "block";

    pictureCell.appendChild(img);
    row.appendChild(pictureCell);

    // --- Name ---
    const nameCell = document.createElement("td");
    nameCell.className = "p-2";
    nameCell.textContent = speaker.name;
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
    const dateCell = document.createElement("td");
    dateCell.className = "p-2";
    dateCell.textContent = speaker.date ?? "";
    row.appendChild(dateCell);

    // --- YouTube icon ---
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

    // Add row to tbody
    tbody.appendChild(row);
  });

  // Add tbody to table
  table.appendChild(tbody);

  // Add table to page
  container.appendChild(table);
}
