// auth.js - Advanced Firebase Authentication Logic

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        if (typeof firebase === 'undefined' || !firebase.auth) {
            showError("Firebase SDK not loaded properly. Ensure you are running via Firebase Hosting.");
            return;
        }

        const auth = firebase.auth();
        const db = firebase.firestore ? firebase.firestore() : null;

        // Check if user is already logged in
        auth.onAuthStateChanged(user => {
            if (user) {
                // Enforce Email Verification (Skip for Google Auth which is auto-verified)
                if (!user.emailVerified) {
                    auth.signOut().then(() => {
                        showError("Please verify your email address before logging in.");
                        if (!window.location.pathname.includes('login.html')) {
                            window.location.href = 'login.html';
                        }
                    });
                    return;
                }

                // User is fully verified and logged in
                if (window.location.pathname.includes('login.html')) {
                    window.location.href = 'index.html';
                }
            } else {
                // Not logged in
                if (!window.location.pathname.includes('login.html') && window.location.pathname.endsWith('.html')) {
                    window.location.href = 'login.html';
                }
            }
        });

        // Set up Event Listeners if we are on the login page
        const formLogin = document.getElementById('form-login');
        const formSignup = document.getElementById('form-signup');
        const btnGoogles = document.querySelectorAll('.btn-google-auth');

        if (formLogin) {
            formLogin.addEventListener('submit', (e) => {
                e.preventDefault();
                const identifier = document.getElementById('login-email').value.trim();
                const pass = document.getElementById('login-password').value;
                
                // If it looks like an email, login normally
                if (identifier.includes('@')) {
                    auth.signInWithEmailAndPassword(identifier, pass)
                        .catch(error => showError(error.message));
                } else {
                    // It's a username! Look up the email in Firestore
                    if (!db) return showError("Firestore database is not initialized.");
                    
                    db.collection('usernames').doc(identifier).get().then(doc => {
                        if (doc.exists) {
                            const mappedEmail = doc.data().email;
                            return auth.signInWithEmailAndPassword(mappedEmail, pass);
                        } else {
                            throw new Error("Username not found. Please try again.");
                        }
                    }).catch(error => showError(error.message));
                }
            });
        }

        if (formSignup) {
            formSignup.addEventListener('submit', (e) => {
                e.preventDefault();
                const username = document.getElementById('signup-username').value.trim();
                const email = document.getElementById('signup-email').value.trim();
                const pass = document.getElementById('signup-password').value;
                
                if (username.includes('@')) {
                    return showError("Username cannot contain an '@' symbol.");
                }

                if (!db) return showError("Firestore database is not initialized.");

                // First check if username is already taken
                db.collection('usernames').doc(username).get().then(doc => {
                    if (doc.exists) {
                        throw new Error("Username is already taken. Please choose another.");
                    }
                    // Create the auth account
                    return auth.createUserWithEmailAndPassword(email, pass);
                })
                .then(userCredential => {
                    const user = userCredential.user;
                    
                    // Send email verification
                    user.sendEmailVerification();
                    
                    // Update profile with username
                    user.updateProfile({ displayName: username });
                    
                    // Save username -> email mapping in Firestore
                    return db.collection('usernames').doc(username).set({
                        email: email
                    }).then(() => {
                        // Immediately sign out to enforce verification
                        auth.signOut();
                        switchTab('login'); // Switch UI to login tab
                        showError("Account created! Please check your email inbox to verify your account before logging in.", true);
                    });
                })
                .catch(error => showError(error.message));
            });
        }

        if (btnGoogles.length > 0) {
            btnGoogles.forEach(btn => {
                btn.addEventListener('click', () => {
                    const provider = new firebase.auth.GoogleAuthProvider();
                    auth.signInWithPopup(provider)
                        .catch(error => showError(error.message));
                });
            });
        }
        
        // Handle Logout logic if the logout button exists
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                auth.signOut().then(() => {
                    window.location.href = 'login.html';
                }).catch(error => console.error("Logout failed:", error));
            });
        }
        
        // Display user name on dashboard if the element exists
        const userGreeting = document.getElementById('user-greeting');
        if (userGreeting) {
            auth.onAuthStateChanged(user => {
                if (user && user.emailVerified) {
                    userGreeting.innerText = user.displayName || user.email;
                }
            });
        }
        
    }, 500); 
});

function showError(msg, isSuccess = false) {
    const errDiv = document.getElementById('auth-error');
    if (errDiv) {
        errDiv.innerText = msg;
        errDiv.style.color = isSuccess ? 'var(--accent-blue)' : 'var(--accent-red)';
        errDiv.style.display = 'block';
    } else {
        console.error("Auth Error:", msg);
    }
}
