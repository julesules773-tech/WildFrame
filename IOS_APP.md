# iOS App (Capacitor shell)

Pyrae as a native iOS app. The app is a **remote shell**: the WKWebView loads
`https://pyrae.co` directly, so the user always gets the latest web build and
the app talks to the existing APIs **same-origin — zero CORS / server changes**.

- Web assets dir (config source of truth): `static/`
- Generated Xcode project: `ios/App/`
- Config: `capacitor.config.json`

## What was set up

| Piece | Detail |
|---|---|
| Capacitor | 8.5.0 (`@capacitor/core`, `@capacitor/cli`, `@capacitor/ios`) |
| Bundle ID | `co.pyrae.wildframe` (change in `capacitor.config.json`, then `npx cap sync`) |
| App name | `Pyrae` |
| Remote URL | `https://pyrae.co` (`server.url` in `capacitor.config.json`) |
| Permissions (Info.plist) | `NSLocationWhenInUseUsageDescription`, `NSCameraUsageDescription`, `NSPhotoLibraryUsageDescription` |
| Package manager | Swift Package Manager (no CocoaPods needed on Capacitor 8) |

## Prerequisites (one-time, on a Mac)

1. **Xcode** from the Mac App Store (free, ~12 GB). This also installs
   `xcodebuild`. If the machine only has Command Line Tools, run:
   `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
2. **Apple Developer account** ($99/yr) — required for running on a real
   device and for TestFlight / App Store. Simulator-only builds work without it.
3. Node 22+ (project already on Node 24).

## Build & run (local)

```bash
# after any change to capacitor.config.json or native code:
npx cap sync ios

# open the project in Xcode:
npx cap open ios
```

In Xcode:
1. `ios/App/App.xcworkspace` opens automatically.
2. **Signing & Capabilities** → select `ios/App/App` target → pick your Team.
   (First time: Xcode will ask to create a signing certificate.)
3. Pick a simulator (or your iPhone), press **Run** (⌘R).

The webview will load the live `https://pyrae.co` — no local server needed.

## TestFlight (first real test with others)

1. In Xcode: **Product → Archive** (device destination, not simulator).
2. **Window → Organizer** → your archive → **Distribute App** →
   **App Store Connect** → Upload.
3. In [App Store Connect](https://appstoreconnect.apple.com), create the app
   record (bundle ID `co.pyrae.wildframe`) if prompted, then
   **TestFlight → Internal Testing** → add testers (up to 100, no review).

## App Store submission (later)

- Privacy policy is already live on the site (App Store asks for the URL).
- The app collects location + photos — App Store Connect will ask you to
  describe usage; point at the site's privacy policy.
- **Guideline 4.2 risk:** a pure webview shell can be rejected as "a website
  in an app". Before launch, integrate `@capacitor/camera` (full-screen native
  capture) and `@capacitor/geolocation` so the app has real native features.
  See the follow-up plan.

## Known webview differences (accepted for beta)

- Camera: the current `<input type="file" capture>` opens the iOS system
  action sheet (Take Photo / Photo Library) instead of a full-screen capture
  UI. Works, just less pretty.
- Compass/heading (`watchHeading`): WKWebView blocks `deviceorientation` on
  iOS, so the wind arrow / heading features silently no-op inside the app.

## Update flow

- **Web changes:** ship to `pyrae.co` as usual — the app picks them up on next
  launch. No App Store release needed.
- **Native changes** (config, plugins, Info.plist): edit → `npx cap sync ios`
  → archive → upload to TestFlight / App Store.
