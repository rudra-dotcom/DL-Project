import SwiftUI

@main
struct RepViTClassifierApp: App {
    @AppStorage(AppAppearance.storageKey) private var appearanceMode = AppAppearance.system.rawValue

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(
                    (AppAppearance.launchOverride ?? AppAppearance(rawValue: appearanceMode))?.colorScheme
                )
        }
    }
}
