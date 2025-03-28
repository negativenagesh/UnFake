# src/components/use_cases_section.py

import streamlit as st

def render_use_cases_section():
    """Render the Use Cases section with the same design as described."""
    use_cases_html = """
    <div class="use-cases-section">
        <div class="header">
            <h1 class="title">Use <span class="highlight">Cases</span></h1>
            <div class="underline"></div>
            <p class="subtitle">Applications and implementation of our deepfake detection technology</p>
        </div>
        <div class="use-cases-grid">
            <div class="use-case-card">
                <div class="icon content-platforms">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                        <line x1="8" y1="21" x2="16" y2="21"></line>
                        <line x1="12" y1="17" x2="12" y2="21"></line>
                        <path d="M7 9l4.5 2.5L16 9"></path>
                    </svg>
                </div>
                <h3>Content Platforms</h3>
                <p>Image hosting platforms like Unsplash can integrate our API to automatically screen uploaded content for deepfakes, maintaining integrity and user trust.</p>
                <ul>
                    <li>Automated content screening</li>
                    <li>Integration via RESTful API</li>
                    <li>Customizable confidence thresholds</li>
                </ul>
                <div class="complexity low">Implementation complexity <span>Low</span></div>
            </div>
            <div class="use-case-card">
                <div class="icon news-media">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"></path>
                        <path d="M18 14h-8"></path>
                        <path d="M15 18h-5"></path>
                        <path d="M10 6h8v4h-8z"></path>
                    </svg>
                </div>
                <h3>News & Media</h3>
                <p>News organizations and media outlets can verify the authenticity of images before publication, preventing misinformation and maintaining journalistic integrity.</p>
                <ul>
                    <li>Pre-publication verification</li>
                    <li>Detailed analysis reports</li>
                    <li>Batch processing for archives</li>
                </ul>
                <div class="complexity medium">Implementation complexity <span>Medium</span></div>
            </div>
            <div class="use-case-card">
                <div class="icon social-media">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="18" cy="5" r="3"></circle>
                        <circle cx="6" cy="12" r="3"></circle>
                        <circle cx="18" cy="19" r="3"></circle>
                        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                    </svg>
                </div>
                <h3>Social Media</h3>
                <p>Social platforms can implement real-time deepfake screening to prevent the spread of manipulated content and protect users from misinformation.</p>
                <ul>
                    <li>Real-time analytics at upload</li>
                    <li>Content flagging system</li>
                    <li>High-volume processing</li>
                </ul>
                <div class="complexity high">Implementation complexity <span>High</span></div>
            </div>
            <div class="use-case-card">
                <div class="icon legal-forensics">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1v4M12 17v6M4 8l2 2M18 8l-2 2M2 12h4M18 12h4M12 12m-5 0a5 5 0 1 0 10 0a5 5 0 1 0 -10 0"></path>
                        <circle cx="12" cy="12" r="9"></circle>
                    </svg>
                </div>
                <h3>Legal & Forensics</h3>
                <p>Legal professionals and forensic analysts can verify the authenticity of photographic evidence, ensuring integrity in legal proceedings.</p>
                <ul>
                    <li>Court-admissible reports</li>
                    <li>Detailed manipulation analysis</li>
                    <li>Chain of custody tracking</li>
                </ul>
                <div class="complexity high">Implementation complexity <span>High</span></div>
            </div>
            <div class="use-case-card">
                <div class="icon academic-research">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                    </svg>
                </div>
                <h3>Academic Research</h3>
                <p>Researchers can utilize our technology for media studies, advancing the development of more sophisticated detection methods.</p>
                <ul>
                    <li>API access for research</li>
                    <li>Dataset collaboration</li>
                    <li>Advanced metrics access</li>
                </ul>
                <div class="complexity medium">Implementation complexity <span>Medium</span></div>
            </div>
            <div class="use-case-card">
                <div class="icon authentication-services">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                        <circle cx="12" cy="16" r="1"></circle>
                    </svg>
                </div>
                <h3>Authentication Services</h3>
                <p>Identity verification services can implement our technology to ensure submitted photos are not manipulated.</p>
                <ul>
                    <li>ID verification checks</li>
                    <li>Secure API integration</li>
                    <li>Fraud prevention</li>
                </ul>
                <div class="complexity medium">Implementation complexity <span>Medium</span></div>
            </div>
        </div>
    </div>
    """
    return use_cases_html