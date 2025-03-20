# components/contact_section.py
import streamlit as st

def render_contact_section():
    # Load CSS
    with open("src/styles/contact.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Contact Section HTML
    contact_html = """
    <div class="contact-section">
        <!-- Left Side: Contact Form -->
        <div class="contact-form">
            <h2>Send us a message</h2>
            <form>
                <label for="full-name">Full Name</label>
                <input type="text" id="full-name" placeholder="Your full name" required>

                <label for="email">Email Address</label>
                <input type="email" id="email" placeholder="you@example.com" required>

                <label for="phone">Phone (optional)</label>
                <input type="tel" id="phone" placeholder="+1 (555) 987-6543">

                <label for="organization">Organization</label>
                <input type="text" id="organization" placeholder="Your company or institution">

                <label for="interest">I'm interested in</label>
                <select id="interest">
                    <option value="" disabled selected>Please select</option>
                    <option value="product">Product Inquiry</option>
                    <option value="support">Support</option>
                    <option value="partnership">Partnership</option>
                    <option value="other">Other</option>
                </select>

                <label for="message">Message</label>
                <textarea id="message" placeholder="Tell us about your specific needs or questions" rows="5"></textarea>

                <div class="checkbox-container">
                    <input type="checkbox" id="privacy" required>
                    <label for="privacy" class="checkbox-label">I agree to the privacy policy and consent to being contacted about UNFake services</label>
                </div>

                <button type="submit" class="send-message-btn">Send Message</button>
            </form>
        </div>

        <!-- Right Side: Get in Touch -->
        <div class="get-in-touch">
            <h2>Get in Touch</h2>
            <div class="contact-info">
                <div class="info-item">
                    <span class="icon">📧</span>
                    <p>Email<br><a href="mailto:info@unfake.ai">info@unfake.ai</a></p>
                </div>
                <div class="info-item">
                    <span class="icon">📞</span>
                    <p>Phone<br><a href="tel:+15551234567">+1 (555) 123-4567</a></p>
                </div>
                <div class="info-item">
                    <span class="icon">📍</span>
                    <p>Office<br>100 Innovation Drive<br>San Francisco, CA 94105<br>United States</p>
                </div>
                <div class="info-item">
                    <span class="icon">⏰</span>
                    <p>Business Hours<br>Monday - Friday: 9AM - 5PM PST</p>
                </div>
            </div>
            <div class="social-links">
                <h3>Follow Us</h3>
                <a href="https://twitter.com" target="_blank"><img src="https://img.icons8.com/ios-filled/50/ffffff/twitter.png" alt="Twitter" class="social-icon"></a>
                <a href="https://linkedin.com" target="_blank"><img src="https://img.icons8.com/ios-filled/50/ffffff/linkedin.png" alt="LinkedIn" class="social-icon"></a>
                <a href="https://facebook.com" target="_blank"><img src="https://img.icons8.com/ios-filled/50/ffffff/facebook.png" alt="Facebook" class="social-icon"></a>
            </div>
            <div class="emergency-call">
                <p>Need immediate assistance?<br>Our team is ready to provide a personalized demo with technology.</p>
                <button class="call-now-btn">Call Now</button>
            </div>
        </div>
    </div>
    """
    st.markdown(contact_html, unsafe_allow_html=True)