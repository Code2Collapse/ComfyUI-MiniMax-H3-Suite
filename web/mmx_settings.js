// MiniMax Suite settings (L3.02; design docs/research/L3_settings_secrets_audit.md S9 in the C2C workspace).
// Declared here so they exist when the Suite is installed alone; the backend reads the saved value through
// mmx_utils/_c2c_saved_setting.py. The environment variable H3_AUTO_CONTINUE=0 still switches it off when set.
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "C2C.MiniMax.Settings",
    settings: [
        {
            id: "c2c.minimax.autoContinue",
            name: "Let long H3 jobs queue their next run by themselves",
            category: ["C2C", "MiniMax", "Jobs", "auto-continue"],
            type: "boolean",
            defaultValue: true,
            tooltip: "Off: H3 Job Controller and Subject Track stop after each run, and you queue the next one. "
                + "A node's own auto_continue option must also be on. The environment variable "
                + "H3_AUTO_CONTINUE=0 switches this off when it is set.",
        },
    ],
});
