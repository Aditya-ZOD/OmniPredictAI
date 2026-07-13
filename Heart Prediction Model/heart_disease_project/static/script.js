document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector(".prediction-form");
    const themeToggle = document.getElementById("themeToggle");
    const themeLabel = document.getElementById("themeLabel");

    const applyTheme = (theme) => {
        document.body.classList.toggle("light-mode", theme === "light");
        if (themeToggle) {
            themeToggle.setAttribute("aria-pressed", String(theme === "light"));
        }
        if (themeLabel) {
            themeLabel.textContent = theme === "light" ? "Light" : "Dark";
        }
        localStorage.setItem("heart-theme", theme);
    };

    const savedTheme = localStorage.getItem("heart-theme") || "dark";
    applyTheme(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const nextTheme = document.body.classList.contains("light-mode") ? "dark" : "light";
            applyTheme(nextTheme);
        });
    }

    if (!form) {
        return;
    }

    const fields = form.querySelectorAll("input, select");

    fields.forEach((field) => {
        field.addEventListener("blur", () => validateField(field));
        field.addEventListener("input", () => {
            if (field.classList.contains("invalid")) {
                validateField(field);
            }
        });
    });

    form.addEventListener("submit", (event) => {
        let isValid = true;
        fields.forEach((field) => {
            if (!validateField(field)) {
                isValid = false;
            }
        });

        if (!isValid) {
            event.preventDefault();
        }
    });

    function validateField(field) {
        const value = field.value.trim();
        let valid = true;

        if (field.hasAttribute("required") && value === "") {
            valid = false;
        }

        if (field.type === "number") {
            const min = field.min !== "" ? Number(field.min) : null;
            const max = field.max !== "" ? Number(field.max) : null;
            const numericValue = Number(value);

            if (value !== "" && (!Number.isFinite(numericValue) || (min !== null && numericValue < min) || (max !== null && numericValue > max))) {
                valid = false;
            }
        }

        field.classList.toggle("invalid", !valid);
        return valid;
    }
});
