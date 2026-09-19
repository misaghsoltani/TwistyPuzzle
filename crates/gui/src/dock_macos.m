// Sets the application icon in the macOS Dock at runtime.
#import <AppKit/AppKit.h>

void set_macos_dock_icon(const unsigned char *png_bytes, int len) {
    if (!png_bytes || len <= 0) {
        return;
    }
    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:png_bytes length:len];
        NSImage *image = [[NSImage alloc] initWithData:data];
        if (image) {
            [NSApp setApplicationIconImage:image];
        }
    }
}
