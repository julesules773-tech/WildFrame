/**
 * Pyrae i18n — shared translation system for all pages.
 *
 * Usage (each page):
 *   1. Include <script src="/i18n.js"></script> before </body>
 *   2. Add data-i18n="key" attributes to translatable text
 *   3. Optionally register page-specific translations via PyraeI18n.register({...})
 *   4. Call PyraeI18n.init() after DOM is ready (auto-called if DOMContentLoaded fires)
 */
var PyraeI18n = (function () {
  "use strict";

  var LANG_KEY = "pyrae_lang";
  var META = {
    en: { flag: "\ud83c\uddec\ud83c\udde7", code: "EN" },
    de: { flag: "\ud83c\udde9\ud83c\uddea", code: "DE" },
    fr: { flag: "\ud83c\uddeb\ud83c\uddf7", code: "FR" },
    es: { flag: "\ud83c\uddea\ud83c\uddf8", code: "ES" },
    pl: { flag: "\ud83c\uddf5\ud83c\uddf1", code: "PL" },
  };
  var COUNTRY_LANG = {
    DE: "de", FR: "fr", ES: "es", PL: "pl",
    AT: "de", CH: "de", BE: "fr", LU: "fr",
  };

  // Page-specific translations registered via register()
  var T = {};

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /** Register a page's translations: { en: {...}, de: {...}, fr: {...}, es: {...} } */
  function register(dict) {
    if (!dict) return;
    ["en", "de", "fr", "es", "pl"].forEach(function (lang) {
      if (!dict[lang]) return;
      if (!T[lang]) T[lang] = {};
      Object.keys(dict[lang]).forEach(function (k) {
        T[lang][k] = dict[lang][k];
      });
    });
  }

  /** Apply a language to all [data-i18n] elements and update the picker. */
  function applyLang(lang) {
    _currentLang = lang;
    var dict = T[lang] || T.en || {};
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      if (dict[key] !== undefined) el.textContent = dict[key];
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
      var key = el.getAttribute("data-i18n-placeholder");
      if (dict[key] !== undefined) el.placeholder = dict[key];
    });
    var m = META[lang] || META.en;
    var flagEl = document.getElementById("lang-flag");
    var codeEl = document.getElementById("lang-code");
    if (flagEl) flagEl.textContent = m.flag;
    if (codeEl) codeEl.textContent = m.code;
    document.querySelectorAll(".lang-option").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
    });
    document.documentElement.setAttribute("lang", lang);
  }

  var _currentLang = "en";

  /** Return the translated string for *key* in the current language, falling back to English. */
  function t(key) {
    var dict = T[_currentLang] || T.en || {};
    return dict[key] !== undefined ? dict[key] : (T.en && T.en[key] !== undefined ? T.en[key] : key);
  }

  function setLang(lang) {
    _currentLang = lang;
    localStorage.setItem(LANG_KEY, lang);
    applyLang(lang);
  }

  /** Detect the best language from localStorage → browser → geolocation. */
  function detectLanguage(callback) {
    var saved = localStorage.getItem(LANG_KEY);
    if (saved && T[saved]) {
      applyLang(saved);
      if (callback) callback(saved);
      return;
    }
    var bl = (navigator.language || "").substring(0, 2).toLowerCase();
    if (T[bl]) {
      applyLang(bl);
      if (callback) callback(bl);
    }
    // Geolocation fallback for more accurate country detection
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        function (pos) {
          fetch(
            "https://geocode.maps.co/reverse?lat=" +
              pos.coords.latitude +
              "&lon=" +
              pos.coords.longitude
          )
            .then(function (r) { return r.json(); })
            .then(function (data) {
              var cc = (
                (data.address && data.address.country_code) ||
                ""
              ).toUpperCase();
              var detected = COUNTRY_LANG[cc];
              if (detected && !localStorage.getItem(LANG_KEY)) {
                applyLang(detected);
                if (callback) callback(detected);
              }
            })
            .catch(function () {});
        },
        function () {},
        { timeout: 8000, maximumAge: 600000 }
      );
    }
  }

  /** Set up the language picker dropdown (call after DOM is ready). */
  function initPicker() {
    var btn = document.getElementById("lang-btn");
    var dropdown = document.getElementById("lang-dropdown");
    if (!btn || !dropdown) return;
    // Skip if handlers already attached (e.g. landing page has its own inline system)
    if (btn.dataset.i18nBound) return;
    btn.dataset.i18nBound = "1";

    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = dropdown.classList.contains("hidden");
      dropdown.classList.toggle("hidden");
      btn.setAttribute("aria-expanded", String(open));
    });

    document.addEventListener("click", function () {
      dropdown.classList.add("hidden");
      btn.setAttribute("aria-expanded", "false");
    });

    dropdown.querySelectorAll(".lang-option").forEach(function (opt) {
      opt.addEventListener("click", function (e) {
        e.stopPropagation();
        setLang(opt.getAttribute("data-lang"));
        dropdown.classList.add("hidden");
        btn.setAttribute("aria-expanded", "false");
      });
    });
  }

  /**
   * Initialize: insert picker HTML, detect language, apply translations.
   * Auto-called on DOMContentLoaded if not called manually.
   */
  function init() {
    // Insert language picker into .header-actions or .top-actions
    var actions = document.querySelector(".header-actions") || document.querySelector(".top-actions");
    if (actions && !document.getElementById("lang-picker")) {
      var picker = document.createElement("div");
      picker.className = "lang-picker";
      picker.id = "lang-picker";
      picker.innerHTML =
        '<button class="lang-btn" id="lang-btn" title="Change language" aria-label="Change language" aria-expanded="false">' +
          '<span class="lang-flag" id="lang-flag">\ud83c\uddec\ud83c\udde7</span>' +
          '<span class="lang-code" id="lang-code">EN</span>' +
          '<span class="lang-chevron">\u25be</span>' +
        '</button>' +
        '<div class="lang-dropdown hidden" id="lang-dropdown">' +
          '<button class="lang-option" data-lang="en">\ud83c\uddec\ud83c\udde7 English</button>' +
          '<button class="lang-option" data-lang="de">\ud83c\udde9\ud83c\uddea Deutsch</button>' +
          '<button class="lang-option" data-lang="fr">\ud83c\uddeb\ud83c\uddf7 Fran\u00e7ais</button>' +
          '<button class="lang-option" data-lang="es">\ud83c\uddea\ud83c\uddf8 Espa\u00f1ol</button>' +
          '<button class="lang-option" data-lang="pl">\ud83c\uddf5\ud83c\uddf1 Polski</button>' +
        '</div>';
      // Insert before the first child so it appears before nav links
      actions.insertBefore(picker, actions.firstChild);
    }

    initPicker();
    detectLanguage();
  }

  // Auto-init when DOM is ready (unless init() was already called
  // or the page has its own lang-picker with handlers attached)
  var _inited = false;
  function _autoInit() {
    if (_inited) return;
    // If the page already has a lang-picker with handlers (e.g. landing page
    // with its own inline i18n system), skip auto-init to avoid conflicts.
    var existingBtn = document.getElementById("lang-btn");
    if (existingBtn && existingBtn.dataset.i18nBound) return;
    _inited = true;
    init();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _autoInit);
  } else {
    // DOM already loaded — schedule for next microtask
    setTimeout(_autoInit, 0);
  }

  return {
    register: register,
    applyLang: applyLang,
    setLang: setLang,
    t: t,
    detectLanguage: detectLanguage,
    init: function () { _inited = true; init(); },
    get LANG_KEY() { return LANG_KEY; },
  };
})();
