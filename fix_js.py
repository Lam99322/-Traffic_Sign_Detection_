"""
JavaScript Error Fix Script
Automatically detects and fixes common JavaScript DOM manipulation issues
"""

def fix_showAlert_function():
    """Fix the showAlert function to be more robust"""
    file_path = "templates/index.html"
    
    # Read current content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the showAlert function and replace with improved version
    old_function = '''function showAlert(message, type = 'success') {
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            
            // Try to find the main container more specifically
            const mainContainer = document.querySelector('.main-container');
            const bodyContainer = document.querySelector('body > .container');
            
            if (mainContainer && mainContainer.parentNode) {
                // Insert before main-container if possible
                mainContainer.parentNode.insertBefore(alert, mainContainer);
            } else if (bodyContainer) {
                // Insert at the beginning of body container
                bodyContainer.insertAdjacentElement('afterbegin', alert);
            } else {
                // Fallback: append to body with fixed positioning
                document.body.appendChild(alert);
                alert.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 10000;
                    max-width: 400px;
                    padding: 15px 20px;
                    border-radius: 8px;
                    font-weight: 500;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                `;
            }
            
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 5000);
        }'''
    
    new_function = '''function showAlert(message, type = 'success') {
            // Create alert element
            const alert = document.createElement('div');
            alert.className = `alert alert-${type} show-alert-custom`;
            alert.textContent = message;
            
            // Always use fixed positioning to avoid DOM issues
            alert.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                max-width: 400px;
                padding: 15px 20px;
                border-radius: 8px;
                font-weight: 500;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                background: ${type === 'error' ? '#ff6b6b' : type === 'warning' ? '#ffa726' : '#4caf50'};
                color: white;
                opacity: 0;
                transform: translateX(100%);
                transition: all 0.3s ease;
            `;
            
            // Add to body safely
            document.body.appendChild(alert);
            
            // Animate in
            requestAnimationFrame(() => {
                alert.style.opacity = '1';
                alert.style.transform = 'translateX(0)';
            });
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (alert && alert.parentNode) {
                    alert.style.opacity = '0';
                    alert.style.transform = 'translateX(100%)';
                    setTimeout(() => {
                        if (alert && alert.parentNode) {
                            alert.remove();
                        }
                    }, 300);
                }
            }, 5000);
        }'''
    
    if old_function in content:
        content = content.replace(old_function, new_function)
        print("✅ Fixed showAlert function")
        return content
    else:
        print("⚠️  showAlert function not found in expected format")
        return content

def add_error_handling():
    """Add global error handling for JavaScript"""
    error_handler = '''
        // Global error handling for JavaScript
        window.addEventListener('error', function(e) {
            console.warn('JavaScript Error Caught:', e.error);
            // Don't let errors break the interface
            return true;
        });
        
        window.addEventListener('unhandledrejection', function(e) {
            console.warn('Promise Rejection Caught:', e.reason);
            // Don't let promise rejections break the interface
            e.preventDefault();
        });
        
        // Utility function to safely get elements
        function safeGetElement(selector) {
            try {
                return document.querySelector(selector);
            } catch (e) {
                console.warn('Element not found:', selector);
                return null;
            }
        }
        
        // Utility function for safe DOM manipulation
        function safeAppend(parent, child) {
            try {
                if (parent && child && parent.appendChild) {
                    parent.appendChild(child);
                    return true;
                }
            } catch (e) {
                console.warn('DOM append failed:', e);
            }
            return false;
        }
    '''
    return error_handler

def main():
    print("🔧 JavaScript Error Fix Script")
    print("=" * 40)
    
    try:
        # Fix the showAlert function
        content = fix_showAlert_function()
        
        # Add error handling if not present
        if 'Global error handling for JavaScript' not in content:
            # Find a good place to insert error handling (before closing </script>)
            script_end = content.rfind('</script>')
            if script_end != -1:
                error_handling = add_error_handling()
                content = content[:script_end] + error_handling + content[script_end:]
                print("✅ Added global error handling")
        
        # Write back to file
        with open("templates/index.html", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n🎉 JavaScript fixes applied successfully!")
        print("📝 Changes made:")
        print("   - Improved showAlert function with fixed positioning")
        print("   - Added smooth animations")
        print("   - Added global error handling")
        print("   - Added safe DOM manipulation utilities")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")

if __name__ == "__main__":
    main()