import { workshop_data } from './workshop_data.js';
import { create_confirmed_speaker_table } from '/../../../js/build_tables.js'

// Select container
const container = document.getElementById("confirmed");

create_confirmed_speaker_table(workshop_data, container);
