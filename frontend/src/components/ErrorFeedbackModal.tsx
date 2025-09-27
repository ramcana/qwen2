/**
 * Error Feedback Modal Component
 * Allows users to provide feedback about errors and their resolution
 */

import React, { useState } from "react";
import {
  X,
  Star,
  Send,
  ThumbsUp,
  ThumbsDown,
  MessageSquare,
} from "lucide-react";
import { useErrorReporting } from "../services/errorReporting";

interface ErrorFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  errorId: string;
  errorType: string;
  errorMessage: string;
  fallbackUsed?: boolean;
  resolutionMethod?: string;
}

interface FeedbackFormData {
  rating: number;
  feedback: string;
  helpfulSuggestions: string[];
  wouldRecommendFallback: boolean;
  additionalComments: string;
}

const ErrorFeedbackModal: React.FC<ErrorFeedbackModalProps> = ({
  isOpen,
  onClose,
  errorId,
  errorType,
  errorMessage,
  fallbackUsed = false,
  resolutionMethod,
}) => {
  const { submitFeedback } = useErrorReporting();
  const [formData, setFormData] = useState<FeedbackFormData>({
    rating: 0,
    feedback: "",
    helpfulSuggestions: [],
    wouldRecommendFallback: false,
    additionalComments: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const predefinedSuggestions = [
    "Error message was clear and helpful",
    "Suggested fixes were accurate",
    "Fallback mode worked well",
    "Recovery was automatic and smooth",
    "Technical details were useful",
    "Resolution time was acceptable",
  ];

  const handleRatingChange = (rating: number) => {
    setFormData((prev) => ({ ...prev, rating }));
  };

  const handleSuggestionToggle = (suggestion: string) => {
    setFormData((prev) => ({
      ...prev,
      helpfulSuggestions: prev.helpfulSuggestions.includes(suggestion)
        ? prev.helpfulSuggestions.filter((s) => s !== suggestion)
        : [...prev.helpfulSuggestions, suggestion],
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.rating === 0) {
      alert("Please provide a rating before submitting.");
      return;
    }

    setIsSubmitting(true);

    try {
      await submitFeedback({
        errorId,
        rating: formData.rating as 1 | 2 | 3 | 4 | 5,
        feedback: formData.feedback || formData.additionalComments,
        helpfulSuggestions: formData.helpfulSuggestions,
        wouldRecommendFallback: formData.wouldRecommendFallback,
      });

      setSubmitted(true);
      setTimeout(() => {
        onClose();
        setSubmitted(false);
        setFormData({
          rating: 0,
          feedback: "",
          helpfulSuggestions: [],
          wouldRecommendFallback: false,
          additionalComments: "",
        });
      }, 2000);
    } catch (error) {
      console.error("Failed to submit feedback:", error);
      alert("Failed to submit feedback. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">
            Error Feedback
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {submitted ? (
          /* Success State */
          <div className="p-6 text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <ThumbsUp className="w-8 h-8 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Thank you for your feedback!
            </h3>
            <p className="text-gray-600">
              Your feedback helps us improve the error handling experience.
            </p>
          </div>
        ) : (
          /* Feedback Form */
          <form onSubmit={handleSubmit} className="p-6 space-y-6">
            {/* Error Context */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-medium text-gray-900 mb-2">Error Details</h3>
              <div className="text-sm text-gray-600 space-y-1">
                <p>
                  <span className="font-medium">Type:</span> {errorType}
                </p>
                <p>
                  <span className="font-medium">Message:</span> {errorMessage}
                </p>
                {fallbackUsed && (
                  <p>
                    <span className="font-medium">Fallback:</span> Used
                    alternative processing
                  </p>
                )}
                {resolutionMethod && (
                  <p>
                    <span className="font-medium">Resolution:</span>{" "}
                    {resolutionMethod}
                  </p>
                )}
              </div>
            </div>

            {/* Rating */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                How would you rate the error handling experience?
              </label>
              <div className="flex items-center space-x-2">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    onClick={() => handleRatingChange(star)}
                    className={`p-1 transition-colors ${
                      star <= formData.rating
                        ? "text-yellow-400"
                        : "text-gray-300 hover:text-yellow-300"
                    }`}
                  >
                    <Star className="w-8 h-8 fill-current" />
                  </button>
                ))}
                <span className="ml-3 text-sm text-gray-600">
                  {formData.rating > 0 && (
                    <>
                      {formData.rating === 1 && "Very Poor"}
                      {formData.rating === 2 && "Poor"}
                      {formData.rating === 3 && "Average"}
                      {formData.rating === 4 && "Good"}
                      {formData.rating === 5 && "Excellent"}
                    </>
                  )}
                </span>
              </div>
            </div>

            {/* Helpful Suggestions */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                What was helpful during error resolution? (Select all that
                apply)
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {predefinedSuggestions.map((suggestion, index) => (
                  <label
                    key={index}
                    className="flex items-center space-x-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={formData.helpfulSuggestions.includes(suggestion)}
                      onChange={() => handleSuggestionToggle(suggestion)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{suggestion}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Fallback Recommendation */}
            {fallbackUsed && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Would you recommend the fallback mode to other users?
                </label>
                <div className="flex items-center space-x-6">
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="radio"
                      name="recommendFallback"
                      checked={formData.wouldRecommendFallback === true}
                      onChange={() =>
                        setFormData((prev) => ({
                          ...prev,
                          wouldRecommendFallback: true,
                        }))
                      }
                      className="text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">Yes</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="radio"
                      name="recommendFallback"
                      checked={formData.wouldRecommendFallback === false}
                      onChange={() =>
                        setFormData((prev) => ({
                          ...prev,
                          wouldRecommendFallback: false,
                        }))
                      }
                      className="text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">No</span>
                  </label>
                </div>
              </div>
            )}

            {/* Additional Comments */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Additional Comments (Optional)
              </label>
              <textarea
                value={formData.additionalComments}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    additionalComments: e.target.value,
                  }))
                }
                rows={4}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Share any additional thoughts about the error handling experience..."
              />
            </div>

            {/* Submit Button */}
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting || formData.rating === 0}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Submitting...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Submit Feedback</span>
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default ErrorFeedbackModal;
